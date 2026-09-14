"""Calendar-time rescaling of bar-count parameters.

Two things are being pinned here, and the second matters more than the first.

1. The arithmetic: a daily bar count becomes the bar count spanning the same
   number of days at the new interval.
2. That the *classification* stays complete. The rescaler works from a written
   list of which parameters are bar counts, so a new integer knob added to a
   config and forgotten here would be silently left at its daily value on
   hourly data - the exact bug this module exists to prevent. The last test in
   this file fails when that happens.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tradingagent.agent import AgentConfig
from tradingagent.cross_section import PortfolioRules
from tradingagent.data import synthetic_ohlcv
from tradingagent.engine import BacktestEngine, ExecutionConfig
from tradingagent.optimize import DEFAULT_SEARCH_SPACE, WalkForwardConfig, space_size
from tradingagent.risk import RiskConfig
from tradingagent.timescale import (
    BAR_COUNT_PARAMS,
    CRYPTO_SCALES,
    DAILY,
    SCALE_INVARIANT_PARAMS,
    TimeScale,
    coverage_report,
    rebalance_for_trades_per_day,
    rescale_agent_config,
    rescale_feature_windows,
    rescale_risk_config,
    rescale_search_space,
    rescale_strategy_params,
    rescale_walk_forward,
    throttle,
    trades_per_day,
)


# --------------------------------------------------------------------------- #
# the arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "interval,per_day",
    [("1d", 1.0), ("6h", 4.0), ("1h", 24.0), ("15m", 96.0), ("5m", 288.0), ("1m", 1440.0)],
)
def test_bars_per_day_for_a_24h_market(interval, per_day):
    assert CRYPTO_SCALES[interval].bars_per_day == per_day
    assert CRYPTO_SCALES[interval].bars_per_year == pytest.approx(per_day * 365.0)


def test_an_equity_session_is_not_a_full_day():
    """A 1-hour bar on a 6.5-hour session is 6.5 bars/day, not 24.

    Getting this wrong would rescale a 20-day lookback to 480 bars on a series
    that only has 130 bars in those 20 days.
    """
    equity = TimeScale("1h", calendar_days=252.0, session_hours=6.5)
    assert equity.bars_per_day == pytest.approx(6.5)
    assert equity.from_daily(20) == 130


def test_daily_is_the_identity():
    for n in (1, 5, 14, 63, 200, 730):
        assert DAILY.from_daily(n) == n


def test_a_lookback_keeps_its_calendar_span():
    """Twelve months stays twelve months, not twelve hours."""
    hourly = CRYPTO_SCALES["1h"]
    assert hourly.from_daily(365) == 365 * 24
    assert hourly.days(hourly.from_daily(60)) == pytest.approx(60.0)


def test_zero_is_a_disabled_sentinel_and_survives():
    """``atr_stop_mult=0``/``kelly_lookback=0`` mean *off*, not "zero bars"."""
    hourly = CRYPTO_SCALES["1h"]
    risk = rescale_risk_config(RiskConfig(kelly_lookback=0), hourly)
    assert risk.kelly_lookback == 0
    space = rescale_search_space({"regime_trend": [0, 200]}, hourly)
    assert space["regime_trend"] == [0, 4800]


# --------------------------------------------------------------------------- #
# configuration objects
# --------------------------------------------------------------------------- #
def test_agent_config_rescales_windows_and_the_annualiser():
    hourly = CRYPTO_SCALES["1h"]
    cfg = rescale_agent_config(AgentConfig(), hourly)
    assert cfg.perf_lookback == 120 * 24
    assert cfg.regime_trend == 200 * 24
    assert cfg.signal_smooth == 5 * 24
    # forgetting this is how an hourly run reports a volatility sqrt(24) too low
    assert cfg.periods_per_year == pytest.approx(365 * 24)


def test_agent_config_rescales_nested_strategy_params():
    cfg = AgentConfig(strategy_params={"ema_trend": {"fast": 20, "slow": 100}})
    out = rescale_agent_config(cfg, CRYPTO_SCALES["6h"])
    assert out.strategy_params["ema_trend"] == {"fast": 80, "slow": 400}


def test_risk_config_rescales_every_bar_count():
    six = CRYPTO_SCALES["6h"]
    risk = rescale_risk_config(RiskConfig(), six)
    assert risk.vol_lookback == 120      # 30 days
    assert risk.atr_n == 56              # 14 days
    assert risk.cooldown_bars == 40      # 10 days
    assert risk.reentry_lockout_bars == 12
    # multipliers and rates must NOT move
    assert risk.atr_stop_mult == RiskConfig().atr_stop_mult
    assert risk.target_vol == RiskConfig().target_vol
    assert risk.max_drawdown_stop == RiskConfig().max_drawdown_stop


def test_walk_forward_windows_keep_their_calendar_span():
    """730 hourly bars is a month. Leaving it there would not be a walk-forward."""
    wf = rescale_walk_forward(WalkForwardConfig(), CRYPTO_SCALES["1h"])
    assert wf.train_bars == 730 * 24
    assert wf.test_bars == 182 * 24
    assert wf.embargo_bars == 10 * 24
    # not bar counts: these must be untouched
    assert wf.n_candidates == WalkForwardConfig().n_candidates
    assert wf.top_k == WalkForwardConfig().top_k
    assert wf.seed == WalkForwardConfig().seed


def test_donchian_exit_channel_stays_inside_the_entry_channel():
    """An exit channel >= the entry channel never triggers, so the floor matters."""
    out = rescale_strategy_params("donchian", {"entry": 3, "exit": 3}, DAILY)
    assert out["exit"] < out["entry"]


def test_ema_fast_stays_faster_than_slow():
    out = rescale_strategy_params("ema_trend", {"fast": 100, "slow": 100}, DAILY)
    assert out["fast"] < out["slow"]


# --------------------------------------------------------------------------- #
# search space
# --------------------------------------------------------------------------- #
def test_rescaling_preserves_the_size_of_the_search_space():
    """The multiple-testing count must not change with the interval.

    If coarse intervals collapsed duplicate values, their deflated Sharpe would
    be corrected for a smaller search than the fine intervals' - and the
    intervals would no longer be comparable on the statistic that decides
    whether any of this is real.
    """
    for scale in CRYPTO_SCALES.values():
        rescaled = rescale_search_space(DEFAULT_SEARCH_SPACE, scale)
        assert set(rescaled) == set(DEFAULT_SEARCH_SPACE)
        assert space_size(rescaled) == space_size(DEFAULT_SEARCH_SPACE)
        for key in DEFAULT_SEARCH_SPACE:
            assert len(rescaled[key]) == len(DEFAULT_SEARCH_SPACE[key])


def test_rescaling_leaves_non_bar_axes_alone():
    rescaled = rescale_search_space(DEFAULT_SEARCH_SPACE, CRYPTO_SCALES["1h"])
    for key in ("target_vol", "max_leverage", "atr_stop_mult", "min_trade_frac", "weighting"):
        assert rescaled[key] == list(DEFAULT_SEARCH_SPACE[key]), key


def test_feature_windows_rescale_to_the_same_months():
    windows = rescale_feature_windows(CRYPTO_SCALES["1h"])
    assert windows["year"] == 252 * 24
    assert windows["skip"] == 21 * 24
    assert windows["trend"] == 200 * 24


# --------------------------------------------------------------------------- #
# trading frequency
# --------------------------------------------------------------------------- #
def test_throttle_holds_the_book_between_rebalances():
    w = pd.Series(np.arange(10.0), index=pd.date_range("2020-01-01", periods=10, tz="UTC"))
    held = throttle(w, 3)
    assert list(held) == [0, 0, 0, 3, 3, 3, 6, 6, 6, 9]


def test_throttle_of_one_is_the_identity():
    w = pd.Series(np.arange(5.0), index=pd.date_range("2020-01-01", periods=5, tz="UTC"))
    pd.testing.assert_series_equal(throttle(w, 1), w)


def test_throttle_cannot_see_the_future():
    """A held weight is a decision already made; poisoning later bars must not
    change any earlier one."""
    idx = pd.date_range("2020-01-01", periods=40, tz="UTC")
    clean = pd.Series(np.arange(40.0), index=idx)
    dirty = clean.copy()
    dirty.iloc[20:] = -999.0
    for every in (1, 3, 7):
        a, b = throttle(clean, every), throttle(dirty, every)
        pd.testing.assert_series_equal(a.iloc[:20], b.iloc[:20], rtol=0, atol=0)


def test_throttle_never_delays_the_first_decision():
    """Bar 0 is always a rebalance bar, so a throttled run enters when the
    unthrottled one does - otherwise slower configs would look better purely by
    sitting out the first few bars."""
    idx = pd.date_range("2020-01-01", periods=12, tz="UTC")
    w = pd.Series(1.0, index=idx)
    assert throttle(w, 5).iloc[0] == 1.0


@pytest.mark.parametrize("interval,every,per_day", [("1h", 1, 24.0), ("1h", 24, 1.0), ("6h", 2, 2.0)])
def test_trades_per_day_arithmetic(interval, every, per_day):
    assert trades_per_day(every, CRYPTO_SCALES[interval]) == pytest.approx(per_day)


def test_rebalance_for_trades_per_day_round_trips():
    hourly = CRYPTO_SCALES["1h"]
    for target in (1.0, 2.0, 4.0, 8.0, 24.0):
        every = rebalance_for_trades_per_day(target, hourly)
        assert trades_per_day(every, hourly) == pytest.approx(target, rel=0.2)


# --------------------------------------------------------------------------- #
# data quality
# --------------------------------------------------------------------------- #
def test_coverage_report_spots_a_hole():
    idx = pd.date_range("2020-01-01", periods=100, freq="1h", tz="UTC")
    frame = pd.DataFrame({"close": 1.0}, index=idx)
    full = coverage_report(frame, CRYPTO_SCALES["1h"])
    assert full["coverage"] == pytest.approx(1.0, rel=0.02)
    assert full["largest_gap_bars"] == pytest.approx(1.0)

    holed = frame.drop(frame.index[40:60])
    gappy = coverage_report(holed, CRYPTO_SCALES["1h"])
    assert gappy["coverage"] < 0.85
    assert gappy["largest_gap_bars"] == pytest.approx(21.0)


# --------------------------------------------------------------------------- #
# the classification guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cls", [AgentConfig, RiskConfig, WalkForwardConfig, PortfolioRules, ExecutionConfig]
)
def test_every_integer_knob_is_classified(cls):
    """No unclassified integer parameter may exist in a config dataclass.

    An integer knob that is neither declared a bar count nor declared
    scale-invariant would keep its daily value on hourly data, and nothing else
    in the test suite would notice.
    """
    known = set(SCALE_INVARIANT_PARAMS)
    for group in BAR_COUNT_PARAMS.values():
        known.update(group)
    unclassified = [
        f.name
        for f in dataclasses.fields(cls)
        if f.type in ("int", int) and f.name not in known
    ]
    assert not unclassified, (
        f"{cls.__name__} has integer parameters that timescale.py does not classify: "
        f"{unclassified}. Add each to BAR_COUNT_PARAMS or SCALE_INVARIANT_PARAMS."
    )
