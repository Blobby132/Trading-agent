"""The agent: an ensemble of strategies with causal, adaptive weights.

No single edge survives every regime, so the agent runs a panel of them and
keeps re-deciding how much to listen to each one. Concretely, at every bar it

1. asks each strategy for the weight it wants,
2. scores each strategy on its own trailing, hypothetical P&L (information
   available at that bar and no later),
3. blends the signals with a softmax of those scores,
4. optionally vetoes signals that fight the long-term trend,
5. hands the blend to the risk layer, which converts a direction into a size.

The scoring is the part that makes this an *agent* rather than a fixed rule:
the mix it trades next month is chosen by what has been working, not by a
constant set by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import indicators as ind
from .engine import BacktestEngine, BacktestResult, ExecutionConfig
from .risk import RiskConfig, apply_sizing
from .strategies import DEFAULT_ENSEMBLE, Strategy, build_ensemble


@dataclass
class AgentConfig:
    """How the ensemble is assembled and blended."""

    strategies: List[str] = field(default_factory=lambda: list(DEFAULT_ENSEMBLE))
    strategy_params: Dict[str, dict] = field(default_factory=dict)
    weighting: str = "adaptive"      # "adaptive" | "equal" | "best"
    perf_lookback: int = 120         # bars of trailing P&L used to score a strategy
    softmax_temp: float = 4.0        # 0 -> equal weights, large -> winner takes all
    drop_negative: bool = True       # ignore strategies whose trailing edge is negative
    regime_filter: bool = True       # veto signals that fight the long trend
    regime_trend: int = 200
    allow_short: bool = True
    signal_smooth: int = 5           # bars of smoothing on the blended signal;
                                     # damps the day-to-day flip-flopping that
                                     # turns into pure transaction cost
    min_active_share: float = 0.35   # see `signal`: floor on the conviction divisor
    periods_per_year: float = 365.0


class TradingAgent:
    """Blend a panel of strategies, then size the blend."""

    def __init__(self, config: AgentConfig | None = None, risk: RiskConfig | None = None):
        self.config = config or AgentConfig()
        self.risk = risk or RiskConfig()
        self.strategies: List[Strategy] = build_ensemble(
            self.config.strategies, self.config.strategy_params
        )
        self.diagnostics_: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ #
    def component_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ppy = self.config.periods_per_year
        cols = {s.name: s.target_weight(df, periods_per_year=ppy) for s in self.strategies}
        return pd.DataFrame(cols, index=df.index).fillna(0.0)

    def component_pnl(self, df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Hypothetical per-bar P&L of each strategy, at unit exposure."""
        asset_ret = df["close"].pct_change().fillna(0.0)
        return signals.shift(1).fillna(0.0).mul(asset_ret, axis=0)

    def blend_weights(self, pnl: pd.DataFrame) -> pd.DataFrame:
        """Softmax of trailing risk-adjusted P&L, per bar."""
        cfg = self.config
        n = len(pnl.columns)
        if cfg.weighting == "equal" or n == 1:
            return pd.DataFrame(1.0 / n, index=pnl.index, columns=pnl.columns)

        lb = max(int(cfg.perf_lookback), 10)
        mean = pnl.rolling(lb, min_periods=lb // 3).mean()
        std = pnl.rolling(lb, min_periods=lb // 3).std(ddof=0).replace(0.0, np.nan)
        score = (mean / std).fillna(0.0)

        if cfg.weighting == "best":
            top = score.eq(score.max(axis=1), axis=0)
            w = top.astype(float)
            return w.div(w.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(1.0 / n)

        z = cfg.softmax_temp * score
        z = z.sub(z.max(axis=1), axis=0)
        w = np.exp(z)
        if cfg.drop_negative:
            # a strategy with a negative trailing edge gets no capital at all,
            # unless every strategy is negative (then fall back to equal weights)
            w = w.where(score > 0, 0.0)
            all_off = w.sum(axis=1) <= 0
            w.loc[all_off] = 1.0 / n
        w = w.div(w.sum(axis=1).replace(0.0, np.nan), axis=0)
        # before there is enough history to judge anyone, weight equally
        warmup = score.abs().sum(axis=1) == 0
        w.loc[warmup] = 1.0 / n
        return w.fillna(1.0 / n)

    # ------------------------------------------------------------------ #
    def signal(self, df: pd.DataFrame) -> pd.Series:
        """Blended target weight in ``[-1, 1]``, decided at each bar's close."""
        cfg = self.config
        signals = self.component_signals(df)
        pnl = self.component_pnl(df, signals)
        weights = self.blend_weights(pnl)

        # Normalise by the weight of the strategies that actually have a view.
        # Without this, a six-model panel in which two models are flat would
        # scale a unanimous signal down to a third of full size purely because
        # the other four had nothing to say. The floor stops a single lone
        # strategy from claiming the whole book.
        active = (signals.abs() > 1e-9).astype(float)
        denom = (weights * active).sum(axis=1).clip(lower=float(cfg.min_active_share))
        combined = (signals * weights).sum(axis=1) / denom

        if cfg.signal_smooth > 1:
            combined = combined.rolling(int(cfg.signal_smooth), min_periods=1).mean()
        if not cfg.allow_short:
            combined = combined.clip(lower=0.0)
        if cfg.regime_filter and cfg.regime_trend > 0:
            trend = ind.ema(df["close"], int(cfg.regime_trend))
            above = df["close"] > trend
            # only take longs above the long-term trend and shorts below it
            combined = combined.where(~(above & (combined < 0)), 0.0)
            combined = combined.where(~((~above) & (combined > 0)), 0.0)

        combined = combined.clip(-1.0, 1.0).fillna(0.0)
        self.diagnostics_ = {
            "signals": signals,
            "component_pnl": pnl,
            "blend_weights": weights,
            "combined": combined.to_frame("combined"),
        }
        return combined.rename("signal")

    def sized_weight(self, df: pd.DataFrame) -> pd.Series:
        """Signal after the risk layer: the leveraged weight the engine trades."""
        raw = self.signal(df)
        strat_ret = raw.shift(1).fillna(0.0) * df["close"].pct_change().fillna(0.0)
        return apply_sizing(
            df,
            raw,
            self.risk,
            periods_per_year=self.config.periods_per_year,
            strategy_returns=strat_ret,
        )

    def backtest(self, df: pd.DataFrame, exec_config: ExecutionConfig | None = None) -> BacktestResult:
        ec = exec_config or ExecutionConfig(periods_per_year=self.config.periods_per_year)
        engine = BacktestEngine(ec, self.risk)
        result = engine.run(df, self.sized_weight(df))
        result.meta["agent_config"] = self.config
        return result


# --------------------------------------------------------------------------- #
# multi-asset
# --------------------------------------------------------------------------- #
class PortfolioAgent:
    """Run one :class:`TradingAgent` per symbol and share one pot of capital.

    Capital is split by inverse volatility (so a wild alt-coin does not silently
    dominate a portfolio nominally split evenly), and the gross book is capped
    by the engine.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        config: AgentConfig | None = None,
        risk: RiskConfig | None = None,
        *,
        allocation: str = "inverse_vol",   # "inverse_vol" | "equal" | "momentum"
        max_positions: int = 0,            # 0 = no cap
        vol_lookback: int = 30,
    ):
        self.symbols = list(symbols)
        self.config = config or AgentConfig()
        self.risk = risk or RiskConfig()
        self.allocation = allocation
        self.max_positions = int(max_positions)
        self.vol_lookback = int(vol_lookback)
        self.agents = {s: TradingAgent(self.config, self.risk) for s in self.symbols}

    def target_weights(self, panel: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        index = panel[self.symbols[0]].index
        sized = pd.DataFrame(
            {s: self.agents[s].sized_weight(panel[s]) for s in self.symbols}, index=index
        ).fillna(0.0)

        if self.allocation == "equal":
            alloc = pd.DataFrame(1.0 / len(self.symbols), index=index, columns=self.symbols)
        elif self.allocation == "momentum":
            mom = pd.DataFrame(
                {s: ind.roc(panel[s]["close"], 60) for s in self.symbols}, index=index
            ).fillna(0.0)
            pos = mom.clip(lower=0.0)
            alloc = pos.div(pos.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(
                1.0 / len(self.symbols)
            )
        else:  # inverse volatility
            vol = pd.DataFrame(
                {
                    s: ind.realized_vol(
                        panel[s]["close"], self.vol_lookback, self.config.periods_per_year
                    )
                    for s in self.symbols
                },
                index=index,
            )
            inv = (1.0 / vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
            alloc = inv.div(inv.sum(axis=1), axis=0).fillna(1.0 / len(self.symbols))

        if self.max_positions and self.max_positions < len(self.symbols):
            rank = sized.abs().rank(axis=1, ascending=False, method="first")
            alloc = alloc.where(rank <= self.max_positions, 0.0)
            alloc = alloc.div(alloc.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

        return (sized * alloc).fillna(0.0)

    def backtest(
        self, panel: Dict[str, pd.DataFrame], exec_config: ExecutionConfig | None = None
    ) -> BacktestResult:
        ec = exec_config or ExecutionConfig(periods_per_year=self.config.periods_per_year)
        engine = BacktestEngine(ec, self.risk)
        result = engine.run(panel, self.target_weights(panel))
        result.meta["agent_config"] = self.config
        return result
