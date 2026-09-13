"""Learning applied to risk instead of to signal.

Part two measured the thing this module exists to act on: a search that chose
*which factor has an edge* lost to a fixed momentum rule, because three years of
noisy data cannot settle that question. But the same three years settle a
different set of questions perfectly well.

| Estimable from a few years | Not estimable from a few years |
|---|---|
| volatility, correlation, beta | which factor has an edge |
| transaction costs, capacity | the sign of a weak signal |
| position sizing, risk budgets | whether this regime is different |

So the signal here is **fixed** - whatever ranker you hand in - and everything
learned is about size: how much of each name to hold, how much of the book to
run, and when to stand down. Each layer is separately switchable, because the
point is to find out which of them pays rather than to ship a bundle.

Every estimate is trailing and lagged one bar, so the size used today never
depends on today's outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .universe import Panel


@dataclass
class RiskLearnerConfig:
    """Which risk layers are on, and how much history each one learns from."""

    inverse_vol: bool = True          # size each name by its own trailing vol
    vol_lookback: int = 63
    correlation_scaling: bool = True  # shrink the book when holdings move together
    corr_lookback: int = 126
    portfolio_vol_target: float = 0.0  # 0 disables; else annualised target
    portfolio_vol_lookback: int = 63
    drawdown_guard: float = 0.0        # 0 disables; else start cutting past this drawdown
    drawdown_floor: float = 0.25       # never cut below this fraction of the book
    max_scale: float = 2.0             # cap on any single multiplier
    min_names: int = 3


def inverse_vol_weights(
    weights: pd.DataFrame, close: pd.DataFrame, *, lookback: int = 63, max_ratio: float = 5.0
) -> pd.DataFrame:
    """Re-split each bar's book by inverse trailing volatility.

    Equal weights are equal *dollars*, not equal risk: a 60%-vol name and a
    15%-vol name at the same weight contribute four times the risk apart. This
    keeps the gross book identical and only changes how it is divided, so it is
    a pure risk-allocation change with nothing else moving.
    """
    rets = np.log(close).diff()
    vol = rets.rolling(lookback, min_periods=max(10, lookback // 3)).std(ddof=0).shift(1)
    inv = 1.0 / vol.replace(0.0, np.nan)
    # bound the ratio so one becalmed name cannot absorb the whole book
    inv = inv.div(inv.median(axis=1), axis=0).clip(upper=max_ratio)
    tilted = weights * inv.reindex_like(weights).fillna(1.0)
    # preserve the original gross exposure per side
    gross_before = weights.abs().sum(axis=1)
    gross_after = tilted.abs().sum(axis=1).replace(0.0, np.nan)
    return tilted.mul((gross_before / gross_after).fillna(0.0), axis=0)


def correlation_scale(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    *,
    lookback: int = 126,
    max_scale: float = 2.0,
) -> pd.Series:
    """Shrink the book when its holdings stop diversifying each other.

    Twelve names that all move together is one position wearing twelve tickers.
    This compares the book's realised volatility against what it would be if the
    same holdings were independent; the ratio is the diversification actually
    being achieved, and the book is scaled by its inverse.

    Computed from realised portfolio returns rather than a covariance matrix -
    with a hundred names and a few hundred bars, a full covariance estimate is
    mostly noise, and this needs only the one number that matters.
    """
    rets = close.pct_change()
    lagged = weights.shift(1)
    port_ret = (lagged * rets).sum(axis=1)

    port_vol = port_ret.rolling(lookback, min_periods=lookback // 3).std(ddof=0)
    # the volatility this book would have if every holding were independent
    name_vol = rets.rolling(lookback, min_periods=lookback // 3).std(ddof=0)
    independent = np.sqrt(((lagged * name_vol) ** 2).sum(axis=1))

    diversification = (independent / port_vol.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    # 1.0 means no diversification at all; higher is better. Scale relative to
    # a typical long-only equity book, which achieves roughly 2x.
    scale = (diversification / 2.0).clip(upper=max_scale)
    return scale.shift(1).fillna(1.0)


def portfolio_vol_scale(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    *,
    target_vol: float,
    lookback: int = 63,
    periods_per_year: float = 252.0,
    max_scale: float = 2.0,
) -> pd.Series:
    """Scale the book toward a constant realised volatility."""
    if target_vol <= 0:
        return pd.Series(1.0, index=weights.index)
    rets = close.pct_change()
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    realised = port_ret.rolling(lookback, min_periods=lookback // 3).std(ddof=0) * np.sqrt(
        periods_per_year
    )
    return (target_vol / realised.replace(0.0, np.nan)).shift(1).clip(upper=max_scale).fillna(1.0)


def drawdown_scale(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    *,
    threshold: float,
    floor: float = 0.25,
) -> pd.Series:
    """Cut size smoothly as the book's own equity falls from its high.

    A hard kill switch is a cliff: one bar either side of it is the difference
    between fully invested and flat, and it latches badly (this repo has the
    scar). Scaling linearly from ``threshold`` down to ``floor`` de-risks into a
    drawdown without ever fully switching off, so recovery does not require a
    separate re-entry rule.

    The equity path is the *hypothetical unlevered* one built from the weights,
    which is available at every bar without running the engine first.
    """
    if threshold <= 0:
        return pd.Series(1.0, index=weights.index)
    rets = close.pct_change()
    port_ret = (weights.shift(1) * rets).sum(axis=1).fillna(0.0)
    equity = (1.0 + port_ret).cumprod()
    dd = equity / equity.cummax() - 1.0
    # 1.0 above the threshold, falling linearly to `floor` at twice the threshold
    excess = (-dd - threshold).clip(lower=0.0) / threshold
    return (1.0 - (1.0 - floor) * excess.clip(upper=1.0)).shift(1).fillna(1.0)


class RiskLearner:
    """Apply the learned risk layers to a fixed signal's weights."""

    def __init__(self, config: RiskLearnerConfig | None = None):
        self.config = config or RiskLearnerConfig()
        self.diagnostics_: Dict[str, pd.Series] = {}

    def apply(
        self, weights: pd.DataFrame, panel: Panel, *, periods_per_year: float = 252.0
    ) -> pd.DataFrame:
        cfg = self.config
        close = panel.close
        out = weights.copy()

        if cfg.inverse_vol:
            out = inverse_vol_weights(out, close, lookback=cfg.vol_lookback)

        scale = pd.Series(1.0, index=out.index)
        if cfg.correlation_scaling:
            corr = correlation_scale(out, close, lookback=cfg.corr_lookback,
                                     max_scale=cfg.max_scale)
            self.diagnostics_["correlation_scale"] = corr
            scale = scale * corr
        if cfg.portfolio_vol_target > 0:
            vol = portfolio_vol_scale(
                out, close, target_vol=cfg.portfolio_vol_target,
                lookback=cfg.portfolio_vol_lookback, periods_per_year=periods_per_year,
                max_scale=cfg.max_scale,
            )
            self.diagnostics_["vol_scale"] = vol
            scale = scale * vol
        if cfg.drawdown_guard > 0:
            guard = drawdown_scale(out, close, threshold=cfg.drawdown_guard,
                                   floor=cfg.drawdown_floor)
            self.diagnostics_["drawdown_scale"] = guard
            scale = scale * guard

        scale = scale.clip(upper=cfg.max_scale).fillna(1.0)
        self.diagnostics_["total_scale"] = scale
        return out.mul(scale, axis=0)
