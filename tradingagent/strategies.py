"""Signal generators.

A strategy maps an OHLCV frame to a *target weight* series in ``[-1, 1]``,
where the value at bar ``t`` is the exposure the strategy wants to hold during
bar ``t + 1``. The engine does the shifting, so a strategy is free to use the
close of bar ``t`` - and must never use anything after it.

Each strategy exposes :meth:`param_space`, a small grid the walk-forward
optimiser searches over.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from . import indicators as ind


class Strategy(ABC):
    """Base class: parameters in the constructor, signal in ``target_weight``."""

    name: str = "strategy"

    def __init__(self, **params):
        defaults = dict(self.defaults())
        unknown = set(params) - set(defaults)
        if unknown:
            raise TypeError(f"{type(self).__name__} got unexpected params: {sorted(unknown)}")
        defaults.update(params)
        self.params: Dict[str, float] = defaults

    # -- interface ---------------------------------------------------------- #
    @staticmethod
    def defaults() -> Dict[str, float]:
        return {}

    @classmethod
    def param_space(cls) -> Dict[str, Sequence]:
        return {k: [v] for k, v in cls.defaults().items()}

    @abstractmethod
    def target_weight(self, df: pd.DataFrame, *, periods_per_year: float = 365.0) -> pd.Series:
        ...

    # -- helpers ------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        inner = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{type(self).__name__}({inner})"

    def clone(self, **overrides) -> "Strategy":
        p = dict(self.params)
        p.update(overrides)
        return type(self)(**p)

    @staticmethod
    def _finish(w: pd.Series, index: pd.Index) -> pd.Series:
        return w.reindex(index).fillna(0.0).clip(-1.0, 1.0).astype(float)


class BuyAndHold(Strategy):
    """Benchmark: fully invested, always."""

    name = "buy_and_hold"

    def target_weight(self, df: pd.DataFrame, *, periods_per_year: float = 365.0) -> pd.Series:
        return pd.Series(1.0, index=df.index)


class EmaTrend(Strategy):
    """Trend following on an EMA spread, gated by ADX and scaled by its size."""

    name = "ema_trend"

    @staticmethod
    def defaults():
        return {"fast": 20, "slow": 100, "adx_n": 14, "adx_min": 18.0, "allow_short": 1, "scale": 1.5}

    @classmethod
    def param_space(cls):
        return {
            "fast": [10, 20, 30, 50],
            "slow": [60, 100, 150, 200],
            "adx_min": [0.0, 15.0, 20.0, 25.0],
            "allow_short": [0, 1],
            "scale": [1.0, 1.5, 2.5],
        }

    def target_weight(self, df: pd.DataFrame, *, periods_per_year: float = 365.0) -> pd.Series:
        p = self.params
        fast, slow = int(p["fast"]), int(p["slow"])
        if fast >= slow:
            return pd.Series(0.0, index=df.index)
        c = df["close"]
        spread = (ind.ema(c, fast) - ind.ema(c, slow)) / c
        # normalise the spread by its own trailing scale so the signal is
        # comparable across assets and volatility regimes
        scale = spread.abs().rolling(slow, min_periods=slow // 2).mean().replace(0.0, np.nan)
        w = np.tanh(float(p["scale"]) * spread / scale)
        strength = ind.adx(df["high"], df["low"], c, int(p["adx_n"]))
        w = w.where(strength >= float(p["adx_min"]), 0.0)
        if not int(p["allow_short"]):
            w = w.clip(lower=0.0)
        return self._finish(w, df.index)


class DonchianBreakout(Strategy):
    """Turtle-style channel breakout with a separate, shorter exit channel."""

    name = "donchian"

    @staticmethod
    def defaults():
        return {"entry": 40, "exit": 15, "allow_short": 1, "trend_filter": 200}

    @classmethod
    def param_space(cls):
        return {
            "entry": [20, 30, 40, 55, 80],
            "exit": [10, 15, 20, 30],
            "allow_short": [0, 1],
            "trend_filter": [0, 100, 200],
        }

    def target_weight(self, df: pd.DataFrame, *, periods_per_year: float = 365.0) -> pd.Series:
        p = self.params
        entry_n, exit_n = int(p["entry"]), int(p["exit"])
        c, h, l = df["close"], df["high"], df["low"]
        entry_ch = ind.donchian(h, l, entry_n)
        exit_ch = ind.donchian(h, l, exit_n)

        long_in = c > entry_ch["upper"]
        long_out = c < exit_ch["lower"]
        short_in = c < entry_ch["lower"]
        short_out = c > exit_ch["upper"]

        tf = int(p["trend_filter"])
        if tf:
            trend = ind.ema(c, tf)
            long_in &= c > trend
            short_in &= c < trend

        state = np.zeros(len(df))
        pos = 0.0
        li, lo = long_in.to_numpy(), long_out.to_numpy()
        si, so = short_in.to_numpy(), short_out.to_numpy()
        allow_short = bool(int(p["allow_short"]))
        for i in range(len(df)):
            if pos > 0 and lo[i]:
                pos = 0.0
            elif pos < 0 and so[i]:
                pos = 0.0
            if pos == 0.0:
                if li[i]:
                    pos = 1.0
                elif allow_short and si[i]:
                    pos = -1.0
            state[i] = pos
        return self._finish(pd.Series(state, index=df.index), df.index)


class MeanReversionZ(Strategy):
    """Fade stretched prices, but only when the market is *not* trending."""

    name = "mean_reversion"

    @staticmethod
    def defaults():
        return {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "adx_max": 20.0, "allow_short": 1}

    @classmethod
    def param_space(cls):
        return {
            "lookback": [10, 20, 30, 50],
            "entry_z": [1.5, 2.0, 2.5],
            "exit_z": [0.0, 0.5, 1.0],
            "adx_max": [15.0, 20.0, 25.0, 100.0],
            "allow_short": [0, 1],
        }

    def target_weight(self, df: pd.DataFrame, *, periods_per_year: float = 365.0) -> pd.Series:
        p = self.params
        z = ind.zscore(df["close"], int(p["lookback"]))
        strength = ind.adx(df["high"], df["low"], df["close"], 14)
        calm = strength <= float(p["adx_max"])
        entry_z, exit_z = float(p["entry_z"]), float(p["exit_z"])
        allow_short = bool(int(p["allow_short"]))

        zz = z.to_numpy()
        ok = calm.to_numpy()
        state = np.zeros(len(df))
        pos = 0.0
        for i in range(len(df)):
            zi = zz[i]
            if np.isnan(zi):
                state[i] = 0.0
                continue
            if pos > 0 and zi >= -exit_z:
                pos = 0.0
            elif pos < 0 and zi <= exit_z:
                pos = 0.0
            if pos == 0.0 and ok[i]:
                if zi <= -entry_z:
                    pos = 1.0
                elif allow_short and zi >= entry_z:
                    pos = -1.0
            state[i] = pos
        return self._finish(pd.Series(state, index=df.index), df.index)


class RsiPullback(Strategy):
    """Buy dips inside an uptrend; exit when the bounce completes."""

    name = "rsi_pullback"

    @staticmethod
    def defaults():
        return {"rsi_n": 14, "buy_below": 35.0, "exit_above": 55.0, "trend": 100}

    @classmethod
    def param_space(cls):
        return {
            "rsi_n": [7, 14, 21],
            "buy_below": [25.0, 30.0, 35.0, 40.0],
            "exit_above": [50.0, 55.0, 65.0],
            "trend": [50, 100, 200],
        }

    def target_weight(self, df: pd.DataFrame, *, periods_per_year: float = 365.0) -> pd.Series:
        p = self.params
        c = df["close"]
        r = ind.rsi(c, int(p["rsi_n"]))
        up = c > ind.ema(c, int(p["trend"]))
        buy = (r < float(p["buy_below"])) & up
        sell = r > float(p["exit_above"])

        state = np.zeros(len(df))
        pos = 0.0
        b, s = buy.to_numpy(), sell.to_numpy()
        for i in range(len(df)):
            if pos > 0 and s[i]:
                pos = 0.0
            elif pos == 0.0 and b[i]:
                pos = 1.0
            state[i] = pos
        return self._finish(pd.Series(state, index=df.index), df.index)


class TimeSeriesMomentum(Strategy):
    """Sign of trailing return, scaled by how unusual that return is."""

    name = "ts_momentum"

    @staticmethod
    def defaults():
        return {"lookback": 60, "smooth": 5, "allow_short": 1, "rank_n": 250}

    @classmethod
    def param_space(cls):
        return {
            "lookback": [20, 40, 60, 120, 200],
            "smooth": [1, 5, 10],
            "allow_short": [0, 1],
            "rank_n": [120, 250, 500],
        }

    def target_weight(self, df: pd.DataFrame, *, periods_per_year: float = 365.0) -> pd.Series:
        p = self.params
        c = df["close"]
        mom = ind.roc(c, int(p["lookback"]))
        if int(p["smooth"]) > 1:
            mom = mom.rolling(int(p["smooth"]), min_periods=1).mean()
        # Direction comes from the sign of the trailing return; size comes from
        # how extreme that return is against its own history, so an ordinary
        # drift gets a small position and an unusual one gets a large position.
        rank = ind.percentile_rank(mom, int(p["rank_n"]))
        w = np.sign(mom) * (2.0 * rank - 1.0).abs()
        if not int(p["allow_short"]):
            w = w.clip(lower=0.0)
        return self._finish(w, df.index)


class BollingerBreakout(Strategy):
    """Ride expansions out of a squeeze; flat once price falls back to the mean."""

    name = "bollinger_breakout"

    @staticmethod
    def defaults():
        return {"n": 20, "k": 2.0, "allow_short": 1, "squeeze_pct": 0.5}

    @classmethod
    def param_space(cls):
        return {
            "n": [15, 20, 30, 50],
            "k": [1.5, 2.0, 2.5],
            "allow_short": [0, 1],
            "squeeze_pct": [0.3, 0.5, 1.0],
        }

    def target_weight(self, df: pd.DataFrame, *, periods_per_year: float = 365.0) -> pd.Series:
        p = self.params
        n, k = int(p["n"]), float(p["k"])
        c = df["close"]
        bands = ind.bollinger(c, n, k)
        width = (bands["upper"] - bands["lower"]) / bands["mid"].replace(0.0, np.nan)
        squeezed = ind.percentile_rank(width, 4 * n) <= float(p["squeeze_pct"])

        up = (c > bands["upper"]) & squeezed
        dn = (c < bands["lower"]) & squeezed
        above_mid = c > bands["mid"]

        state = np.zeros(len(df))
        pos = 0.0
        u, d, am = up.fillna(False).to_numpy(), dn.fillna(False).to_numpy(), above_mid.to_numpy()
        allow_short = bool(int(p["allow_short"]))
        for i in range(len(df)):
            if pos > 0 and not am[i]:
                pos = 0.0
            elif pos < 0 and am[i]:
                pos = 0.0
            if pos == 0.0:
                if u[i]:
                    pos = 1.0
                elif allow_short and d[i]:
                    pos = -1.0
            state[i] = pos
        return self._finish(pd.Series(state, index=df.index), df.index)


STRATEGY_REGISTRY: Dict[str, type] = {
    cls.name: cls
    for cls in (
        BuyAndHold,
        EmaTrend,
        DonchianBreakout,
        MeanReversionZ,
        RsiPullback,
        TimeSeriesMomentum,
        BollingerBreakout,
    )
}

#: The ensemble the agent uses by default - two trend models, one breakout,
#: one momentum model and two counter-trend models, so the mix has something
#: to say in both trending and ranging markets.
DEFAULT_ENSEMBLE: List[str] = [
    "ema_trend",
    "donchian",
    "ts_momentum",
    "bollinger_breakout",
    "mean_reversion",
    "rsi_pullback",
]


def make_strategy(name: str, **params) -> Strategy:
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](**params)


def build_ensemble(names: Sequence[str] | None = None, params: Dict[str, dict] | None = None) -> List[Strategy]:
    names = list(names or DEFAULT_ENSEMBLE)
    params = params or {}
    return [make_strategy(n, **params.get(n, {})) for n in names]
