"""Causal technical indicators.

Every function here uses only past and current observations: no centred
windows, no ``shift(-1)``, no full-sample statistics. ``tests/test_causality.py``
enforces that by feeding a truncated series and checking the tail is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def roc(s: pd.Series, n: int) -> pd.Series:
    """Rate of change over ``n`` bars."""
    return s.pct_change(n)


def log_returns(s: pd.Series) -> pd.Series:
    return np.log(s).diff()


def zscore(s: pd.Series, n: int) -> pd.Series:
    mean = s.rolling(n, min_periods=n).mean()
    std = s.rolling(n, min_periods=n).std(ddof=0)
    return (s - mean) / std.replace(0.0, np.nan)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's average true range."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(avg_loss.notna() | avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd, "sd": sd})


def donchian(high: pd.Series, low: pd.Series, n: int = 20) -> pd.DataFrame:
    """Channel of the previous ``n`` completed bars (current bar excluded)."""
    upper = high.shift(1).rolling(n, min_periods=n).max()
    lower = low.shift(1).rolling(n, min_periods=n).min()
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2.0})


def realized_vol(close: pd.Series, n: int = 20, periods_per_year: float = 365.0) -> pd.Series:
    """Annualised standard deviation of log returns."""
    r = log_returns(close)
    return r.rolling(n, min_periods=max(2, n // 2)).std(ddof=0) * np.sqrt(periods_per_year)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's ADX - how strongly the market is trending, direction-agnostic."""
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(high, low, close)
    atr_n = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1.0 / n, adjust=False, min_periods=n
    ).mean() / atr_n.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1.0 / n, adjust=False, min_periods=n
    ).mean() / atr_n.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rolling_drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown of ``equity`` against its running maximum (0 to -1)."""
    peak = equity.cummax()
    return equity / peak - 1.0


def rolling_sharpe(returns: pd.Series, n: int, periods_per_year: float) -> pd.Series:
    mean = returns.rolling(n, min_periods=max(5, n // 3)).mean()
    std = returns.rolling(n, min_periods=max(5, n // 3)).std(ddof=0)
    return (mean / std.replace(0.0, np.nan)) * np.sqrt(periods_per_year)


def percentile_rank(s: pd.Series, n: int) -> pd.Series:
    """Where the current value sits inside its own trailing window, in [0, 1]."""
    return s.rolling(n, min_periods=max(5, n // 3)).rank(pct=True)


def feature_frame(
    df: pd.DataFrame, *, periods_per_year: float = 365.0, prefix: str = ""
) -> pd.DataFrame:
    """Standard feature set used by the bundled strategies."""
    c, h, l = df["close"], df["high"], df["low"]
    feats = pd.DataFrame(index=df.index)
    feats[f"{prefix}ret"] = log_returns(c)
    for n in (10, 20, 50, 100, 200):
        feats[f"{prefix}ema{n}"] = ema(c, n)
    feats[f"{prefix}rsi14"] = rsi(c, 14)
    feats[f"{prefix}atr14"] = atr(h, l, c, 14)
    feats[f"{prefix}atr_pct"] = feats[f"{prefix}atr14"] / c
    feats[f"{prefix}adx14"] = adx(h, l, c, 14)
    feats[f"{prefix}vol20"] = realized_vol(c, 20, periods_per_year)
    feats[f"{prefix}vol60"] = realized_vol(c, 60, periods_per_year)
    feats[f"{prefix}mom20"] = roc(c, 20)
    feats[f"{prefix}mom60"] = roc(c, 60)
    feats[f"{prefix}z20"] = zscore(c, 20)
    return feats
