"""OHLCV data loading.

Four sources, all returning the same tidy frame:

    index : pandas.DatetimeIndex (UTC, ascending, unique)
    cols  : open, high, low, close, volume

* :func:`load_coinbase`  - Coinbase Exchange public candles (no API key, crypto).
* :func:`load_yahoo`     - yfinance (stocks, ETFs, FX, crypto). Optional import.
* :func:`load_csv`       - a local CSV that was downloaded elsewhere.
* :func:`synthetic_ohlcv`- regime-switching simulator used by the test-suite
                           and for sanity checks when the network is unavailable.

:func:`load_prices` is the front door and caches to ``data/cache`` so a Colab
runtime does not re-download on every execution.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Coinbase Exchange supports only this discrete set of candle widths (seconds).
COINBASE_GRANULARITIES = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}

# Bars per calendar year, used to annualise statistics.
BARS_PER_YEAR = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "1h": 365 * 24,
    "6h": 365 * 4,
    "1d": 365,
    "1wk": 52,
}

DEFAULT_CACHE_DIR = os.environ.get(
    "TRADING_AGENT_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache"),
)


def bars_per_year(interval: str, *, calendar_days: int = 365) -> float:
    """Annualisation factor for ``interval``.

    Crypto trades every day, so the default calendar is 365 days. Pass
    ``calendar_days=252`` for daily equity data.
    """
    if interval in BARS_PER_YEAR:
        base = BARS_PER_YEAR[interval]
        if interval == "1d":
            return float(calendar_days)
        return float(base) * (calendar_days / 365.0) if calendar_days != 365 else float(base)
    raise KeyError(f"unknown interval {interval!r}; known: {sorted(BARS_PER_YEAR)}")


# --------------------------------------------------------------------------- #
# normalisation helpers
# --------------------------------------------------------------------------- #
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame into the canonical shape."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    rename = {"adj close": "close", "adj_close": "close", "vol": "volume", "time": "timestamp"}
    out = out.rename(columns=rename)

    if not isinstance(out.index, pd.DatetimeIndex):
        for cand in ("timestamp", "date", "datetime"):
            if cand in out.columns:
                out = out.set_index(pd.to_datetime(out[cand], utc=True)).drop(columns=[cand])
                break
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("could not find a datetime index or column")

    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")

    out = out[OHLCV_COLUMNS].astype(float)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["open", "high", "low", "close"])

    # A bar whose high/low do not bracket open/close is corrupt; repair rather
    # than drop, so a single bad print does not punch a hole in the series.
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)
    out = out[out["close"] > 0]
    out.index.name = "timestamp"
    return out


# --------------------------------------------------------------------------- #
# Coinbase Exchange
# --------------------------------------------------------------------------- #
def load_coinbase(
    product_id: str = "BTC-USD",
    interval: str = "1d",
    start: str | _dt.datetime = "2017-01-01",
    end: Optional[str | _dt.datetime] = None,
    *,
    max_requests: int = 400,
    pause: float = 0.22,
    session=None,
) -> pd.DataFrame:
    """Download candles from the Coinbase Exchange public REST API.

    The endpoint returns at most 300 candles per call, so this walks backwards
    from ``end`` until it reaches ``start`` or runs out of history.
    """
    import requests  # imported lazily so the package works offline

    if interval not in COINBASE_GRANULARITIES:
        raise ValueError(
            f"interval {interval!r} unsupported by Coinbase; use one of {sorted(COINBASE_GRANULARITIES)}"
        )
    gran = COINBASE_GRANULARITIES[interval]
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end is not None else pd.Timestamp.utcnow().tz_localize("UTC")

    sess = session or requests.Session()
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    frames: list[pd.DataFrame] = []
    cursor = end_ts
    span = pd.Timedelta(seconds=gran * 300)

    for _ in range(max_requests):
        if cursor <= start_ts:
            break
        window_start = max(start_ts, cursor - span)
        params = {
            "granularity": gran,
            "start": window_start.isoformat().replace("+00:00", "Z"),
            "end": cursor.isoformat().replace("+00:00", "Z"),
        }
        resp = sess.get(url, params=params, timeout=30)
        if resp.status_code == 429:  # courtesy back-off on rate limit
            time.sleep(1.0)
            continue
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        # Coinbase order: [time, low, high, open, close, volume]
        chunk = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        chunk["timestamp"] = pd.to_datetime(chunk["time"], unit="s", utc=True)
        frames.append(chunk.drop(columns=["time"]))
        cursor = window_start
        time.sleep(pause)

    if not frames:
        raise RuntimeError(f"Coinbase returned no candles for {product_id} {interval}")

    df = _normalize(pd.concat(frames, ignore_index=True))
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


# --------------------------------------------------------------------------- #
# Yahoo Finance
# --------------------------------------------------------------------------- #
def load_yahoo(
    symbol: str = "BTC-USD",
    interval: str = "1d",
    start: str = "2017-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download candles with yfinance (stocks, ETFs, FX and crypto)."""
    import yfinance as yf

    raw = yf.download(
        symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=True
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"yfinance returned nothing for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):  # single-symbol download still nests
        raw.columns = raw.columns.get_level_values(0)
    return _normalize(raw)


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV from a CSV with a date/timestamp column or index."""
    return _normalize(pd.read_csv(path))


# --------------------------------------------------------------------------- #
# synthetic data
# --------------------------------------------------------------------------- #
def synthetic_ohlcv(
    n: int = 2000,
    *,
    seed: int = 7,
    interval: str = "1d",
    start: str = "2018-01-01",
    s0: float = 100.0,
    drift: float = 0.35,
    vol: float = 0.65,
    trend_persistence: float = 0.985,
) -> pd.DataFrame:
    """Regime-switching GBM with intrabar high/low.

    Not a substitute for real data - it exists so the engine, the strategies
    and the walk-forward machinery can be exercised deterministically and
    without a network connection.
    """
    rng = np.random.default_rng(seed)
    ppy = bars_per_year(interval)
    dt = 1.0 / ppy
    mu_state = 0.0
    closes = np.empty(n)
    opens = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)
    price = s0
    for i in range(n):
        # slowly mean-reverting drift state creates trends and chop by turns
        mu_state = trend_persistence * mu_state + rng.normal(0.0, 1.0 - trend_persistence)
        mu = drift * mu_state
        shock = rng.normal(mu * dt, vol * np.sqrt(dt))
        open_p = price
        close_p = max(price * float(np.exp(shock)), 1e-8)
        wick = abs(rng.normal(0.0, vol * np.sqrt(dt) * 0.7))
        high_p = max(open_p, close_p) * (1.0 + wick)
        low_p = min(open_p, close_p) * (1.0 - wick)
        opens[i], closes[i], highs[i], lows[i] = open_p, close_p, high_p, low_p
        price = close_p

    freq = {"1d": "1D", "1h": "1h", "6h": "6h", "15m": "15min", "5m": "5min", "1m": "1min"}[interval]
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    vols = rng.lognormal(mean=10.0, sigma=0.5, size=n)
    return _normalize(
        pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}, index=idx
        )
    )


# --------------------------------------------------------------------------- #
# front door + cache
# --------------------------------------------------------------------------- #
def _cache_path(source: str, symbol: str, interval: str, start, end, cache_dir: str) -> str:
    key = f"{source}|{symbol}|{interval}|{start}|{end}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    safe = symbol.replace("/", "-").replace(" ", "")
    return os.path.join(cache_dir, f"{source}_{safe}_{interval}_{digest}.csv")


def load_prices(
    symbol: str = "BTC-USD",
    interval: str = "1d",
    start: str = "2017-01-01",
    end: Optional[str] = None,
    *,
    source: str = "coinbase",
    cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Load one symbol from ``source``, caching the result on disk."""
    path = _cache_path(source, symbol, interval, start, end, cache_dir)
    if cache and not refresh and os.path.exists(path):
        return _normalize(pd.read_csv(path))

    if source == "coinbase":
        df = load_coinbase(symbol, interval=interval, start=start, end=end, **kwargs)
    elif source in ("yahoo", "yfinance"):
        df = load_yahoo(symbol, interval=interval, start=start, end=end, **kwargs)
    elif source == "csv":
        df = load_csv(kwargs.pop("path", symbol))
    elif source == "synthetic":
        df = synthetic_ohlcv(interval=interval, start=start, **kwargs)
    else:
        raise ValueError(f"unknown source {source!r}")

    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(path)
    return df


def load_universe(
    symbols: Sequence[str],
    interval: str = "1d",
    start: str = "2017-01-01",
    end: Optional[str] = None,
    *,
    source: str = "coinbase",
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Load several symbols; symbols that fail to download are skipped."""
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym] = load_prices(sym, interval=interval, start=start, end=end, source=source, **kwargs)
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the run
            print(f"[data] skipping {sym}: {type(exc).__name__}: {exc}")
    if not out:
        raise RuntimeError("no symbols could be loaded")
    return out


def align_universe(universe: Dict[str, pd.DataFrame], how: str = "inner") -> Dict[str, pd.DataFrame]:
    """Restrict every symbol to a common timestamp index."""
    idx = None
    for df in universe.values():
        idx = df.index if idx is None else (idx.intersection(df.index) if how == "inner" else idx.union(df.index))
    return {k: v.reindex(idx).ffill().dropna() for k, v in universe.items()}
