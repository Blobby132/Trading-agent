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
    signal_deadband: float = 0.0     # zero any blended signal weaker than this.
                                     # 0.0 reproduces the historical behaviour
                                     # exactly; see `signal` for the evidence.
    signal_shape: float = 1.0        # exponent applied to |signal| before sizing.
                                     # 1.0 is the historical linear mapping.
    trend_tilt: float = 0.0          # tilt exposure toward stronger trends. 0 = off.
    accel_tilt: float = 0.0          # tilt exposure AWAY from recently accelerated
                                     # trends. 0 = off.
    tilt_lookback: int = 90          # bars defining "trend strength"
    tilt_rank_window: int = 365      # trailing window the rank percentile is taken over
    trend_floor: float = 0.0         # minimum long exposure while the long-horizon
                                     # trend is up. 0.0 reproduces history.
    trend_floor_lookback: int = 252  # bars defining "the long-horizon trend"
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

        # -- exposure shaping ------------------------------------------- #
        # Both transforms are pointwise and monotone in the already-causal
        # blended signal, so neither can introduce look-ahead; the poisoning
        # tests cover them anyway.
        #
        # The blended signal's magnitude is the panel's *agreement*, not a
        # forecast of size. Measured on BTC 2017-2026, bars where agreement was
        # near-total (|signal| > 0.75) returned +33.0 bps on the next bar
        # against +16.3 unconditional (t = 2.45), while bars of faint agreement
        # (0.01 < |signal| < 0.25) returned -17.3 bps. Faint agreement is the
        # panel disagreeing, and disagreement precedes chop.
        #
        # `signal_deadband` stands aside rather than taking a token position in
        # that faint-agreement region. `signal_shape` > 1 bends capital toward
        # agreement without raising the cap - it can only ever reduce |signal|,
        # so it cannot smuggle in leverage.
        if float(cfg.signal_deadband) > 0.0:
            combined = combined.where(combined.abs() >= float(cfg.signal_deadband), 0.0)
        if float(cfg.signal_shape) != 1.0:
            combined = np.sign(combined) * combined.abs() ** float(cfg.signal_shape)

        # -- exposure tilt -------------------------------------------------- #
        # Two conditioning variables that survived a causal screen on BTC
        # 2017-2026 (8 candidates tested, 2 s.e. bar):
        #
        #   trend strength - the trailing 90-bar return's percentile rank. Top
        #   quartile returned +45.2 bps on the next bar against -1.4 in the
        #   bottom (t = 2.46).
        #
        #   acceleration - 63-bar return minus 126-bar return, ranked. Its sign
        #   is the OPPOSITE of the obvious guess: top-quartile acceleration
        #   returned -6.2 bps against +35.3 in the bottom (t = -2.49). A double
        #   sort shows why it is not redundant with strength: inside the
        #   strongest trend quartile, steady momentum earned +57.2 bps and
        #   recently-accelerated momentum -15.6. Steady trends persist;
        #   parabolic ones mean-revert.
        #
        # Both ranks are uniform by construction, so a tilt of the form
        # 1 +/- k(2r - 1) has an expected multiplier of 1 - it REDISTRIBUTES
        # exposure across states rather than adding any. Combined with the
        # [-1, 1] clip below, it cannot raise the leverage cap.
        #
        # Applied BEFORE the floor, so the floor remains a hard minimum and the
        # protected component keeps its guarantee.
        tilt_k, tilt_j = float(cfg.trend_tilt), float(cfg.accel_tilt)
        if tilt_k != 0.0 or tilt_j != 0.0:
            lb, win = int(cfg.tilt_lookback), int(cfg.tilt_rank_window)
            close = df["close"]
            multiplier = pd.Series(1.0, index=combined.index)
            if tilt_k != 0.0:
                strength = close.pct_change(lb)
                rank = strength.rolling(win, min_periods=max(lb, win // 3)).rank(pct=True)
                multiplier = multiplier + tilt_k * (2.0 * rank.reindex(combined.index) - 1.0)
            if tilt_j != 0.0:
                # acceleration: the recent window's return against twice that
                # window's, so a trend that has sped up scores high
                accel = close.pct_change(lb) - close.pct_change(2 * lb)
                arank = accel.rolling(win, min_periods=max(lb, win // 3)).rank(pct=True)
                multiplier = multiplier - tilt_j * (2.0 * arank.reindex(combined.index) - 1.0)
            # a missing rank during warm-up means "no opinion", not "go flat"
            combined = combined * multiplier.fillna(1.0).clip(lower=0.0)

        # -- trend floor -------------------------------------------------- #
        # Measured against a plain 12-month momentum rule on BTC 2017-2026, the
        # bars where the rule was invested and this agent was flat are 21.9% of
        # the sample and the asset returned +40.2% annualised across them. That
        # block is the bull-market upside the panel leaves behind: its component
        # models are short-horizon state machines that stand aside on any pause,
        # while the twelve-month trend is still intact underneath.
        #
        # So while the long-horizon trend is up, do not go all the way flat.
        # This is a FLOOR, not a signal - the panel can still size above it, and
        # its defensive behaviour in a downtrend is untouched because the floor
        # simply does not apply there. Only the long side is floored, and only
        # where the panel is not already asking to be short, so this can never
        # flip a direction.
        floor = float(cfg.trend_floor)
        if floor > 0.0 and int(cfg.trend_floor_lookback) > 0:
            trailing = df["close"].pct_change(int(cfg.trend_floor_lookback))
            up = (trailing > 0).reindex(combined.index).fillna(False)
            liftable = up & (combined >= 0.0) & (combined < floor)
            combined = combined.where(~liftable, floor)

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
