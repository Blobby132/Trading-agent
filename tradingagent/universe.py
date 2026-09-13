"""Universes and the wide panel representation cross-sectional work needs.

Single-asset code is happiest with one tidy OHLCV frame per symbol. Ranking a
hundred names against each other every day is the opposite: you want one wide
frame per *field*, with dates down and symbols across, so a cross-sectional
operation is a single row-wise call instead of a loop.

:class:`Panel` holds that representation and converts back to the per-symbol
frames the execution engine consumes.

A note on survivorship
----------------------
``US_LARGE_CAP`` is a hand-written list of names that are liquid *today*. Testing
it back to 2005 is survivorship-biased: companies that were large in 2005 and
then collapsed are missing, so a long-only backtest over it is flattered.

Two things blunt that, neither of which removes it:

* the list deliberately includes names that were large and then did badly
  (INTC, WBA, BA, GE, F, PFE, T, VZ, CVS, PARA, NKE), rather than only winners;
* a *cross-sectional* strategy is far less exposed than a long-only one, because
  the bias lifts every name in the universe roughly equally and a ranking model
  only trades the differences between them.

For a bias-free study you need point-in-time index membership, which no free
data source provides. Treat long-only results here as an upper bound.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data import OHLCV_COLUMNS, load_prices

#: ~100 liquid US large caps across every sector, chosen for long history.
#: Includes conspicuous laggards on purpose - see the survivorship note above.
US_LARGE_CAP: List[str] = [
    # technology & communications
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE", "CRM", "AMD", "INTC",
    "TXN", "QCOM", "IBM", "ACN", "ADI", "MU", "INTU", "NOW", "AMAT", "LRCX",
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "PARA",
    # consumer
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "GM", "F",
    "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "WBA",
    # healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "GILD", "BMY", "MDT", "SYK", "BDX", "CVS", "ZTS",
    # financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SPGI",
    "SCHW", "C", "CB", "PNC", "AIG", "MET", "TRV", "ICE",
    # industrials & materials
    "CAT", "DE", "HON", "UPS", "RTX", "LMT", "GD", "BA", "GE", "MMM", "EMR",
    "ITW", "NSC", "FDX", "ROP", "LIN", "APD", "SHW", "NEM",
    # energy & utilities
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "OXY", "KMI", "WMB",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL",
    # real estate
    "PLD", "AMT", "PSA", "SPG", "O",
]

#: Coinbase USD pairs with enough history to rank against one another.
CRYPTO_MAJORS: List[str] = [
    "BTC-USD", "ETH-USD", "LTC-USD", "BCH-USD", "LINK-USD", "XLM-USD", "ADA-USD",
    "DOT-USD", "SOL-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD", "ATOM-USD",
    "ALGO-USD", "XTZ-USD", "FIL-USD", "AAVE-USD", "UNI-USD", "ETC-USD", "EOS-USD",
]

#: A small, fast universe for smoke tests and demos.
US_LARGE_CAP_SMALL: List[str] = [
    "AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "KO", "CAT", "NEE", "WMT",
    "INTC", "BA", "T", "PFE", "DIS",
]

UNIVERSES: Dict[str, List[str]] = {
    "us_large_cap": US_LARGE_CAP,
    "us_large_cap_small": US_LARGE_CAP_SMALL,
    "crypto_majors": CRYPTO_MAJORS,
}


@dataclass
class Panel:
    """Wide OHLCV frames: dates down, symbols across.

    Symbols that had not listed yet - or have stopped trading - are ``NaN`` for
    those dates rather than forward-filled. The execution engine treats a NaN
    price as *not tradeable*, which is how a universe that changes over time is
    meant to behave.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def symbols(self) -> List[str]:
        return list(self.close.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    def __len__(self) -> int:
        return len(self.close)

    def slice(self, window: slice) -> "Panel":
        return Panel(*(getattr(self, f).iloc[window] for f in OHLCV_COLUMNS))

    def loc(self, start=None, end=None) -> "Panel":
        return Panel(*(getattr(self, f).loc[start:end] for f in OHLCV_COLUMNS))

    def tradeable(self) -> pd.DataFrame:
        """True where a bar can actually be traded."""
        ok = self.close.notna() & self.open.notna() & (self.close > 0) & (self.open > 0)
        return ok

    def min_history(self, bars: int) -> "Panel":
        """Drop symbols with fewer than ``bars`` observations in the whole sample.

        **This is a look-ahead filter.** It decides membership using the full
        history, including bars that had not happened yet at the start of a
        backtest, so a name that listed late and then ran for years is kept from
        day one. It is convenient for assembling a universe and it is not
        point-in-time correct.

        Use :meth:`require_history_before` when the distinction matters. The
        engine already handles a name that has not listed - its bars are NaN and
        it simply is not traded - so filtering on total history buys tidiness,
        not correctness.
        """
        keep = [s for s in self.symbols if self.close[s].notna().sum() >= bars]
        return Panel(*(getattr(self, f)[keep] for f in OHLCV_COLUMNS))

    def require_history_before(self, date, bars: int) -> "Panel":
        """Keep only symbols that already had ``bars`` observations by ``date``.

        The point-in-time-correct version of :meth:`min_history`: membership is
        decided using data available at ``date`` and nothing after it, so a
        backtest starting there could have been run with exactly this universe.

        It does not fix the *ticker list* itself - a hand-written list of names
        that are liquid today is survivorship-biased however it is filtered, and
        only point-in-time index membership data fixes that.
        """
        cutoff = pd.Timestamp(date, tz="UTC") if pd.Timestamp(date).tzinfo is None else pd.Timestamp(date)
        history = self.close.loc[:cutoff].notna().sum()
        keep = [s for s in self.symbols if history.get(s, 0) >= bars]
        return Panel(*(getattr(self, f)[keep] for f in OHLCV_COLUMNS))

    def to_frames(self) -> Dict[str, pd.DataFrame]:
        """Per-symbol OHLCV frames, as the execution engine expects."""
        return {
            sym: pd.DataFrame(
                {field: getattr(self, field)[sym] for field in OHLCV_COLUMNS},
                index=self.index,
            )
            for sym in self.symbols
        }

    @classmethod
    def from_frames(cls, frames: Dict[str, pd.DataFrame]) -> "Panel":
        """Build a panel from per-symbol frames, on the union of their dates."""
        if not frames:
            raise ValueError("no frames given")
        index = None
        for df in frames.values():
            index = df.index if index is None else index.union(df.index)
        wide = {
            field: pd.DataFrame(
                {sym: df[field].reindex(index) for sym, df in frames.items()}, index=index
            )
            for field in OHLCV_COLUMNS
        }
        return cls(**wide)

    def describe(self) -> pd.DataFrame:
        """Per-symbol coverage - the first thing to look at after loading."""
        rows = []
        for sym in self.symbols:
            col = self.close[sym].dropna()
            rows.append(
                {
                    "symbol": sym,
                    "bars": len(col),
                    "first": col.index[0].date() if len(col) else None,
                    "last": col.index[-1].date() if len(col) else None,
                }
            )
        return pd.DataFrame(rows).sort_values("bars", ascending=False).reset_index(drop=True)


def load_panel(
    symbols: Sequence[str] | str,
    *,
    interval: str = "1d",
    start: str = "2005-01-01",
    end: Optional[str] = None,
    source: str = "yahoo",
    pause: float = 1.2,
    min_bars: int = 250,
    verbose: bool = True,
    max_consecutive_failures: int = 5,
    **kwargs,
) -> Panel:
    """Download a universe into a :class:`Panel`.

    Symbols that fail to download are skipped with a message rather than killing
    the run - with a hundred names, one delisted ticker should not cost you the
    other ninety-nine. ``pause`` spaces the requests out, which matters because
    Yahoo rate-limits bursts.

    But a *source* that is down is not the same as one bad ticker. After
    ``max_consecutive_failures`` symbols in a row, this gives up and raises, so
    a caller can fall back to another source immediately. Without that, a
    rate-limited Yahoo turns a two-minute download into hours of retry backoff
    that ends with an empty panel anyway.
    """
    names = UNIVERSES[symbols] if isinstance(symbols, str) else list(symbols)
    frames: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []
    consecutive_failures = 0
    session = None
    if source == "yahoo":
        import requests

        session = requests.Session()

    for i, sym in enumerate(names):
        try:
            extra = dict(kwargs)
            if session is not None:
                extra.setdefault("session", session)
            frames[sym] = load_prices(
                sym, interval=interval, start=start, end=end, source=source, **extra
            )
            consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the run
            failed.append(sym)
            consecutive_failures += 1
            if verbose:
                print(f"[universe] skipping {sym}: {type(exc).__name__}: {str(exc)[:70]}")
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    f"{source!r} failed on {consecutive_failures} symbols in a row "
                    f"(last: {sym}: {str(exc)[:80]}) - treating the source as unavailable "
                    f"rather than retrying the remaining {len(names) - i - 1} names"
                ) from exc
        if pause and i < len(names) - 1:
            time.sleep(pause)

    if not frames:
        raise RuntimeError("no symbols loaded")
    panel = Panel.from_frames(frames).min_history(min_bars)
    if verbose:
        print(
            f"[universe] {len(panel.symbols)} symbols, {len(panel):,} dates "
            f"({panel.index[0].date()} -> {panel.index[-1].date()})"
            + (f", {len(failed)} failed" if failed else "")
        )
    return panel
