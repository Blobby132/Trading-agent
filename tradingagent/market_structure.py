"""Is an asset a candidate for this strategy at all?

The strategy is a trend follower. Trend following needs momentum to *continue*,
and whether it does is a property of the market, not of the parameters. Measured
across six assets over 2015-2026:

    1-month momentum autocorrelation   crypto [+0.075, +0.177]  equity [-0.177, -0.084]
    3-month momentum autocorrelation   crypto [+0.147, +0.301]  equity [-0.273, -0.146]
    share of bars in a sideways regime crypto [ 9.7%,  9.9%]    equity [16.5%, 43.5%]
    share of bars in a bear regime     crypto [25.0%, 31.4%]    equity [ 4.3%, 12.6%]

Those ranges are **disjoint**, not merely different, and the momentum figures
differ in sign: momentum continues in crypto and mean-reverts in the equity
indices. That is the structural precondition, and it explains the cross-asset
result that motivated this module - the same frozen configuration beats a
matched-exposure baseline on BTC and ETH and loses to it on SPY, QQQ, DIA and
IWM.

So this is a **screen, not a signal**. Nothing here goes into a trading
decision. It answers "should this asset be considered at all", before any
backtest is run on it, which is the cheapest possible way to avoid discovering
the same failure again on a seventh instrument.

Two warnings the numbers cannot carry themselves:

* **The estimates are weak individually.** Non-overlapping 3-month windows give
  roughly 48 independent observations over a decade, so a single asset's
  autocorrelation carries a standard error near 0.14. The evidence is in the
  *consistency of the sign across six assets*, not in any one figure.
* **Six assets, two of them crypto, one era.** The crypto/equity split could be
  an asset-class coincidence of 2015-2026. A screen that says "this looks like
  crypto" is not the same as one that says "this will work".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

#: Horizons in bars, and the labels used in the report.
MOMENTUM_HORIZONS: Dict[str, int] = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}

#: The boundary observed between the two asset classes. Deliberately set at zero
#: for the autocorrelations rather than at the midpoint of the observed gap: a
#: sign change is a statement about the market's character, while a midpoint
#: would be a threshold fitted to six points.
FAVOURABLE = {
    "ac_1m_above": 0.0,
    "ac_3m_above": 0.0,
    "sideways_share_below": 0.15,
    "bear_share_above": 0.15,
}


@dataclass
class StructureReport:
    """Descriptive statistics for one asset, plus a screen verdict."""

    symbol: str
    stats: Dict[str, float]
    passes: Dict[str, bool]

    @property
    def score(self) -> int:
        """How many of the four structural conditions the asset meets."""
        return int(sum(self.passes.values()))

    @property
    def verdict(self) -> str:
        n = self.score
        if n == 4:
            return "candidate - matches the structure the strategy was built on"
        if n >= 2:
            return "marginal - partial match, expect degraded behaviour"
        return "unsuitable - the structural precondition is absent"

    def describe(self) -> str:
        lines = [f"{self.symbol}: {self.score}/4 conditions - {self.verdict}"]
        for k, v in self.stats.items():
            lines.append(f"    {k:<24} {v:+.4f}")
        return "\n".join(lines)


def momentum_autocorrelation(close: pd.Series, horizon: int) -> float:
    """Correlation between one ``horizon``-bar return and the next one.

    Positive means trends continue at that horizon; negative means they reverse.
    Computed on every bar, so consecutive observations overlap heavily and the
    effective sample is roughly ``len(close) / horizon`` - see the module note
    about standard errors before quoting a single value.
    """
    past = close.pct_change(horizon)
    future = past.shift(-horizon)
    both = pd.concat([past, future], axis=1).dropna()
    if len(both) < horizon * 3:
        return float("nan")
    # A series whose period returns barely vary - a constant-growth path, or a
    # long flat stretch - has no correlation to report, and computing one anyway
    # returns a confident-looking number derived from floating-point noise.
    # Refuse rather than mislead.
    scale = float(np.nanmedian(np.abs(both.to_numpy()))) or 1.0
    if both.iloc[:, 0].std() < 1e-9 * scale or both.iloc[:, 1].std() < 1e-9 * scale:
        return float("nan")
    return float(both.iloc[:, 0].corr(both.iloc[:, 1]))


def effective_observations(n_bars: int, horizon: int) -> int:
    """Independent (non-overlapping) windows behind an autocorrelation estimate."""
    return max(int(n_bars // max(horizon, 1)) - 1, 0)


def autocorrelation_stderr(n_bars: int, horizon: int) -> float:
    """Rough standard error, from the *independent* window count, not the bar count.

    Using the bar count here would understate the error several-fold and make a
    noisy estimate look decisive. This is the number that keeps a single asset's
    reading honest.
    """
    n = effective_observations(n_bars, horizon)
    return float("nan") if n < 2 else 1.0 / np.sqrt(n)


def analyse(
    frame: pd.DataFrame,
    *,
    symbol: str = "asset",
    periods_per_year: float = 365.0,
    favourable: Optional[Dict[str, float]] = None,
) -> StructureReport:
    """Measure the four screening statistics for one OHLCV frame."""
    from .robustness import classify_regimes

    f = {**FAVOURABLE, **(favourable or {})}
    close = frame["close"]
    stats: Dict[str, float] = {}
    for label, h in MOMENTUM_HORIZONS.items():
        stats[f"ac_{label}"] = momentum_autocorrelation(close, h)
        stats[f"ac_{label}_stderr"] = autocorrelation_stderr(len(close), h)

    regimes = classify_regimes(close, periods_per_year=periods_per_year)["trend"]
    occupancy = regimes.value_counts(normalize=True)
    stats["sideways_share"] = float(occupancy.get("sideways", 0.0))
    stats["bear_share"] = float(occupancy.get("bear", 0.0))
    stats["bull_share"] = float(occupancy.get("bull", 0.0))
    stats["ann_vol"] = float(np.log(close).diff().std() * np.sqrt(periods_per_year))

    passes = {
        "ac_1m_positive": bool(stats["ac_1m"] > f["ac_1m_above"]),
        "ac_3m_positive": bool(stats["ac_3m"] > f["ac_3m_above"]),
        "sideways_low": bool(stats["sideways_share"] < f["sideways_share_below"]),
        "bear_present": bool(stats["bear_share"] > f["bear_share_above"]),
    }
    return StructureReport(symbol=symbol, stats=stats, passes=passes)


def compare(
    frames: Dict[str, pd.DataFrame], *, periods_per_year: Dict[str, float] | float = 365.0
) -> pd.DataFrame:
    """Screen several assets at once, strongest structural match first."""
    rows = []
    for symbol, frame in frames.items():
        ppy = (periods_per_year[symbol] if isinstance(periods_per_year, dict)
               else periods_per_year)
        report = analyse(frame, symbol=symbol, periods_per_year=ppy)
        rows.append({"symbol": symbol, "score": report.score, "verdict": report.verdict,
                     **{k: v for k, v in report.stats.items() if not k.endswith("_stderr")},
                     **report.passes})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
