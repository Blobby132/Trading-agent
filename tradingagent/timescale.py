"""Calendar-time rescaling of bar-count parameters.

Every lookback in this package is written as a number of **bars**, and every
default was chosen against **daily** bars. That is a hidden assumption, and it
breaks the moment the candle interval changes: ``mom_lookback=60`` means sixty
days on daily candles and sixty *hours* - two and a half days - on hourly ones.
Re-running a daily-tuned configuration on hourly data without rescaling is not
"the same strategy at a higher frequency", it is a different, much faster
strategy, and any difference in the result says nothing about the interval.

So this module makes the assumption explicit. A :class:`TimeScale` knows how
many bars of a given interval fit in a day, and the ``rescale_*`` functions
convert every bar-count parameter from its daily meaning to the equivalent
calendar span at the new interval.

What rescaling does and does not buy you
----------------------------------------
It buys comparability of *horizon*: a 20-bar EMA on daily data and a 480-bar
EMA on hourly data both average about twenty days of price.

It does **not** make the two identical. The hourly version is built from
intraday prices the daily version never sees, so it reacts to overnight gaps
and microstructure that the daily series smooths away by construction. Two
strategies with the same horizon and different sampling are still two
strategies. Rescaling removes the *trivial* explanation for a performance
difference; it does not remove every explanation.

Three kinds of parameter
------------------------
* **bar counts** - lookbacks, smoothing windows, lockouts. These rescale.
* **dimensionless** - multipliers, thresholds, z-scores, fractions. These do
  not, and silently rescaling one would be a bug.
* **annualised rates** - target volatility, borrow rates. These are already
  expressed per year and are interval-independent by construction; what
  changes is the annualisation factor used to *measure* them, which lives in
  :func:`tradingagent.data.bars_per_year`.

The split is written down in :data:`BAR_COUNT_PARAMS` and
:data:`SCALE_INVARIANT_PARAMS` rather than inferred, because inferring it from
a name is exactly the sort of cleverness that produces a silent mis-scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import BARS_PER_YEAR

#: Bars of each interval that fit in one 24-hour day. Crypto trades around the
#: clock, so this is a clean calendar division; see :class:`TimeScale` for what
#: happens on a session-based (equity) calendar.
BARS_PER_DAY: Dict[str, float] = {
    "1m": 1440.0,
    "5m": 288.0,
    "15m": 96.0,
    "1h": 24.0,
    "6h": 4.0,
    "1d": 1.0,
    "1wk": 1.0 / 7.0,
}

#: Every parameter in the package that is measured in bars, grouped by where it
#: lives. If you add a bar-count parameter anywhere, add it here too - the test
#: suite checks that the config dataclasses hold no unclassified integer knob.
BAR_COUNT_PARAMS: Dict[str, tuple] = {
    "AgentConfig": ("perf_lookback", "regime_trend", "signal_smooth",
                    "trend_floor_lookback", "tilt_lookback", "tilt_rank_window"),
    "RiskConfig": (
        "vol_lookback",
        "atr_n",
        "reentry_lockout_bars",
        "cooldown_bars",
        "kelly_lookback",
        "kelly_min_observations",
    ),
    "WalkForwardConfig": ("train_bars", "test_bars", "embargo_bars"),
    "ema_trend": ("fast", "slow", "adx_n"),
    "donchian": ("entry", "exit", "trend_filter"),
    "mean_reversion": ("lookback",),
    "rsi_pullback": ("rsi_n", "trend"),
    "ts_momentum": ("lookback", "smooth", "rank_n"),
    "bollinger_breakout": ("n",),
    "PortfolioRules": ("rebalance_every",),
    # names as they appear in the optimiser's flat search space
    "search_space": (
        "perf_lookback",
        "regime_trend",
        "signal_smooth",
        "ema_fast",
        "ema_slow",
        "donchian_entry",
        "mom_lookback",
        "rebalance_every",
    ),
}

#: Parameters that must NOT be rescaled, and why. Multipliers and thresholds
#: are dimensionless; rates are already per-year.
SCALE_INVARIANT_PARAMS: Dict[str, str] = {
    "adx_min": "threshold on an index, dimensionless",
    "adx_max": "threshold on an index, dimensionless",
    "scale": "tanh gain, dimensionless",
    "allow_short": "flag",
    "trend_filter_flag": "flag",
    "entry_z": "z-score, dimensionless",
    "exit_z": "z-score, dimensionless",
    "buy_below": "RSI level, dimensionless",
    "exit_above": "RSI level, dimensionless",
    "k": "standard deviations, dimensionless",
    "squeeze_pct": "percentile, dimensionless",
    "softmax_temp": "temperature, dimensionless",
    "min_active_share": "fraction, dimensionless",
    "target_vol": "annualised rate",
    "max_leverage": "ratio",
    "min_leverage": "ratio",
    "atr_stop_mult": "multiple of ATR, dimensionless",
    "take_profit_mult": "multiple of ATR, dimensionless",
    "max_drawdown_stop": "fraction of peak equity",
    "kelly_fraction": "fraction",
    "kelly_cap": "fraction",
    "risk_per_trade": "fraction of equity",
    "min_trade_frac": "fraction of a position",
    "trail_stop": "flag",
    "regime_filter": "flag",
    "top_frac": "fraction of the universe",
    "bottom_frac": "fraction of the universe",
    "gross": "ratio",
    "max_weight": "fraction of equity",
    "min_names": "a count of symbols, not of bars",
    "min_positions": "a count of symbols, not of bars",
    "signal_deadband": "a threshold on a [-1,1] signal, dimensionless",
    "signal_shape": "an exponent, dimensionless",
    "trend_floor": "a fraction of equity, dimensionless",
    "trend_tilt": "a tilt coefficient on a [0,1] rank, dimensionless",
    "accel_tilt": "a tilt coefficient on a [0,1] rank, dimensionless",
    "n_candidates": "a count of configurations, not of bars",
    "top_k": "a count of configurations, not of bars",
    "seed": "an RNG seed",
}

#: Bar counts baked into :func:`tradingagent.features.raw_features`, expressed
#: as the daily-bar windows the module was written with.
FEATURE_WINDOWS_DAILY: Dict[str, int] = {
    "skip": 21,       # the "1" in 12-1: one month skipped for reversal
    "year": 252,      # the "12"
    "half": 126,      # six months
    "quarter": 63,    # three months, also the vol window
    "trend": 200,     # the 200-day moving average in dist_200
    "short_rev": 5,   # one week
    "rsi": 14,
    "liquidity": 21,
}


@dataclass(frozen=True)
class TimeScale:
    """How bars of one interval map onto calendar time.

    ``calendar_days`` is the number of days per year on which the market is
    open: 365 for crypto, ~252 for US equities. It only affects annualisation,
    never the bar-count conversion, because a lookback of "twenty days" means
    twenty *tradeable* days in both worlds.

    ``session_hours`` is how many hours a day the market is open, and it is
    what makes an intraday equity bar different from an intraday crypto bar: a
    1-hour bar on a 6.5-hour equity session is 6.5 bars/day, not 24.
    """

    interval: str = "1d"
    calendar_days: float = 365.0
    session_hours: float = 24.0

    def __post_init__(self) -> None:
        if self.interval not in BARS_PER_DAY:
            raise KeyError(
                f"unknown interval {self.interval!r}; known: {sorted(BARS_PER_DAY)}"
            )

    # -- basic conversions -------------------------------------------------- #
    @property
    def bars_per_day(self) -> float:
        """Bars in one trading day at this interval."""
        if self.interval in ("1d", "1wk"):
            return BARS_PER_DAY[self.interval]
        return BARS_PER_DAY[self.interval] * (self.session_hours / 24.0)

    @property
    def bars_per_year(self) -> float:
        return self.bars_per_day * self.calendar_days

    @property
    def ratio_to_daily(self) -> float:
        """Multiplier taking a daily-bar count to this interval's bar count."""
        return self.bars_per_day / 1.0

    def bars(self, days: float, *, minimum: int = 1) -> int:
        """Bars spanning ``days`` trading days, floored at ``minimum``."""
        return int(max(minimum, round(float(days) * self.bars_per_day)))

    def days(self, bars: float) -> float:
        """Trading days spanned by ``bars`` bars."""
        return float(bars) / self.bars_per_day

    def from_daily(self, daily_bars: float, *, minimum: int = 1) -> int:
        """Convert a bar count written for daily data to this interval."""
        return self.bars(float(daily_bars), minimum=minimum)

    def describe(self) -> str:
        return (
            f"{self.interval}: {self.bars_per_day:g} bars/day, "
            f"{self.bars_per_year:,.0f} bars/yr "
            f"(x{self.ratio_to_daily:g} vs daily)"
        )


DAILY = TimeScale("1d")
#: The crypto intervals this project can actually source (Coinbase Exchange).
CRYPTO_SCALES: Dict[str, TimeScale] = {
    iv: TimeScale(iv, calendar_days=365.0, session_hours=24.0)
    for iv in ("1m", "5m", "15m", "1h", "6h", "1d")
}


# --------------------------------------------------------------------------- #
# rescaling configuration objects
# --------------------------------------------------------------------------- #
def _rescale_mapping(
    params: Mapping[str, object], names: Iterable[str], scale: TimeScale, *, minimum: int = 1
) -> Dict[str, object]:
    out = dict(params)
    for key in names:
        if key in out and out[key] is not None:
            value = float(out[key])
            # zero is a sentinel for "disabled" everywhere in this package
            # (atr_n aside), so it must survive the conversion unchanged
            out[key] = 0 if value == 0 else scale.from_daily(value, minimum=minimum)
    return out


def rescale_strategy_params(name: str, params: Mapping[str, object], scale: TimeScale) -> Dict:
    """Rescale one strategy's bar-count parameters to ``scale``.

    ``donchian``'s exit channel is floored at 2 bars and at less than its own
    entry channel, because an exit channel that meets or exceeds the entry
    channel never triggers.
    """
    out = _rescale_mapping(params, BAR_COUNT_PARAMS.get(name, ()), scale, minimum=2)
    if name == "donchian" and "entry" in out and "exit" in out:
        out["exit"] = max(2, min(int(out["exit"]), int(out["entry"]) - 1))
    if name == "ema_trend" and "fast" in out and "slow" in out:
        out["fast"] = max(2, min(int(out["fast"]), int(out["slow"]) - 1))
    return out


def rescale_agent_config(cfg, scale: TimeScale):
    """Rescale an :class:`~tradingagent.agent.AgentConfig` onto ``scale``.

    Also rescales the per-strategy overrides it carries, and resets
    ``periods_per_year`` so the risk layer annualises against the right number
    of bars - forgetting that is how an hourly backtest ends up reporting a
    volatility 5x too low.
    """
    strategy_params = {
        name: rescale_strategy_params(name, params, scale)
        for name, params in (cfg.strategy_params or {}).items()
    }
    fields = _rescale_mapping(
        {k: getattr(cfg, k) for k in BAR_COUNT_PARAMS["AgentConfig"]},
        BAR_COUNT_PARAMS["AgentConfig"],
        scale,
    )
    return replace(
        cfg,
        strategy_params=strategy_params,
        periods_per_year=scale.bars_per_year,
        **fields,
    )


def rescale_risk_config(cfg, scale: TimeScale):
    """Rescale a :class:`~tradingagent.risk.RiskConfig` onto ``scale``."""
    fields = _rescale_mapping(
        {k: getattr(cfg, k) for k in BAR_COUNT_PARAMS["RiskConfig"]},
        BAR_COUNT_PARAMS["RiskConfig"],
        scale,
    )
    return replace(cfg, **fields)


def rescale_walk_forward(cfg, scale: TimeScale):
    """Rescale a :class:`~tradingagent.optimize.WalkForwardConfig`.

    This is the one that matters most for honesty. ``train_bars=730`` means two
    years on daily data; leaving it at 730 hourly bars would train on a month
    and call it a walk-forward. The windows must stay the same *calendar* span
    or the folds are not comparable across intervals.
    """
    fields = _rescale_mapping(
        {k: getattr(cfg, k) for k in BAR_COUNT_PARAMS["WalkForwardConfig"]},
        BAR_COUNT_PARAMS["WalkForwardConfig"],
        scale,
    )
    return replace(cfg, **fields)


def rescale_search_space(space: Mapping[str, Sequence], scale: TimeScale) -> Dict[str, List]:
    """Rescale every bar-count axis of an optimiser search space.

    The *shape* of the space is preserved exactly - same keys, same number of
    values per key - so the multiple-testing count is unchanged and the
    deflated Sharpe stays comparable across intervals. Only the numbers move.

    Duplicate values can appear after rounding at coarse intervals (a 1-bar and
    a 5-bar smoothing window both become 1 bar on 6-hourly data). They are kept
    rather than de-duplicated: collapsing them would shrink the search space for
    some intervals and not others, and then the deflated Sharpe would be
    comparing searches of different sizes.
    """
    out: Dict[str, List] = {k: list(v) for k, v in space.items()}
    for key in BAR_COUNT_PARAMS["search_space"]:
        if key not in out:
            continue
        out[key] = [
            0 if float(v) == 0 else scale.from_daily(float(v), minimum=1) for v in out[key]
        ]
    return out


def rescale_feature_windows(scale: TimeScale) -> Dict[str, int]:
    """The cross-sectional feature windows, converted to ``scale``.

    ``mom_12_1`` is "twelve months of trend, skipping the last month". On daily
    equity bars that is ``close.shift(21) / close.shift(252)``. On hourly bars
    the same *economic* statement is ``shift(136) / shift(1638)`` - the same
    twelve months, not twelve hours.
    """
    return {
        name: scale.from_daily(daily, minimum=2)
        for name, daily in FEATURE_WINDOWS_DAILY.items()
    }


# --------------------------------------------------------------------------- #
# trading frequency
# --------------------------------------------------------------------------- #
def throttle(weights: pd.Series | pd.DataFrame, every: int) -> pd.Series | pd.DataFrame:
    """Only let the target weight change every ``every`` bars.

    This is how trading frequency is swept independently of signal frequency:
    the strategy still forms a view on every bar, but the book is only allowed
    to move on a rebalance bar and holds its previous target in between.

    Strictly backward-looking - a held weight is a decision already made - so it
    cannot introduce lookahead. Bar 0 is always a rebalance bar, so the throttle
    never delays the first entry relative to the unthrottled series.
    """
    every = int(every)
    if every <= 1:
        return weights
    keep = np.zeros(len(weights), dtype=bool)
    keep[::every] = True
    mask = pd.Series(keep, index=weights.index)
    if isinstance(weights, pd.DataFrame):
        held = weights.where(mask, np.nan).ffill()
    else:
        held = weights.where(mask, np.nan).ffill()
    return held.fillna(0.0)


def trades_per_day(rebalance_every: int, scale: TimeScale) -> float:
    """Upper bound on rebalances per day at this interval and throttle."""
    return scale.bars_per_day / max(int(rebalance_every), 1)


def rebalance_for_trades_per_day(target: float, scale: TimeScale) -> int:
    """The throttle that comes closest to ``target`` rebalances per day."""
    return int(max(1, round(scale.bars_per_day / max(float(target), 1e-9))))


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def rescale_table(
    scales: Sequence[TimeScale], space: Mapping[str, Sequence], wf_daily: Mapping[str, int]
) -> pd.DataFrame:
    """The audit table: every daily bar-count and what it became per interval.

    Printed in the sweep report so the rescaling is checkable by eye rather
    than taken on trust.
    """
    rows: List[dict] = []
    for key in BAR_COUNT_PARAMS["search_space"]:
        if key not in space:
            continue
        row = {"parameter": key, "daily": list(space[key])}
        for sc in scales:
            row[sc.interval] = [
                0 if float(v) == 0 else sc.from_daily(float(v)) for v in space[key]
            ]
        rows.append(row)
    for key, value in wf_daily.items():
        row = {"parameter": f"wf.{key}", "daily": value}
        for sc in scales:
            row[sc.interval] = sc.from_daily(value)
        rows.append(row)
    return pd.DataFrame(rows).set_index("parameter")


def coverage_report(frame: pd.DataFrame, scale: TimeScale) -> Dict[str, float]:
    """How complete a downloaded series actually is at this interval.

    Thin history is the failure mode that makes a high-frequency backtest look
    good for no reason: missing bars during illiquid periods quietly remove the
    hours when a strategy would have been stopped out. This counts what is
    missing rather than assuming the download is dense.
    """
    if frame.empty:
        return {"bars": 0.0, "expected": 0.0, "coverage": 0.0, "largest_gap_bars": 0.0}
    span_days = (frame.index[-1] - frame.index[0]) / pd.Timedelta(days=1)
    expected = max(span_days * scale.bars_per_day, 1.0)
    step = pd.Timedelta(days=1.0 / scale.bars_per_day)
    gaps = frame.index.to_series().diff().dropna()
    largest = float(gaps.max() / step) if len(gaps) else 0.0
    return {
        "bars": float(len(frame)),
        "expected": float(expected),
        "coverage": float(len(frame) / expected),
        "largest_gap_bars": largest,
        "span_days": float(span_days),
    }
