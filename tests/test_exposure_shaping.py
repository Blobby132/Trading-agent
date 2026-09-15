"""Exposure shaping: deadband and convexity.

Two knobs that change how a blended signal becomes a position. Both are
opt-in, both must be exactly neutral at their defaults, and neither may
introduce leverage or look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagent.agent import AgentConfig, TradingAgent
from tradingagent.data import synthetic_ohlcv


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return synthetic_ohlcv(900, seed=17)


def test_defaults_reproduce_the_historical_signal_exactly(data):
    """The knobs are additive. At their defaults every published number must
    still reproduce, or this change silently rewrites the baseline."""
    a = TradingAgent(AgentConfig()).signal(data)
    b = TradingAgent(AgentConfig(signal_deadband=0.0, signal_shape=1.0)).signal(data)
    pd.testing.assert_series_equal(a, b, rtol=0, atol=0)


def test_shaping_can_never_increase_exposure(data):
    """|s|**k for k>1 and |s|<=1 is a contraction. This is what makes the
    transform safe: it cannot smuggle in leverage through the back door."""
    linear = TradingAgent(AgentConfig()).signal(data)
    for k in (1.5, 2.0, 3.0):
        shaped = TradingAgent(AgentConfig(signal_shape=k)).signal(data)
        assert (shaped.abs() <= linear.abs() + 1e-12).all(), k
        assert shaped.abs().max() <= 1.0 + 1e-12


def test_shaping_preserves_direction(data):
    """Only size may change. A transform that flipped a sign would be a
    different strategy, not a shaping of this one."""
    linear = TradingAgent(AgentConfig()).signal(data)
    shaped = TradingAgent(AgentConfig(signal_shape=2.0)).signal(data)
    both = (linear.abs() > 1e-9) & (shaped.abs() > 1e-9)
    assert (np.sign(linear[both]) == np.sign(shaped[both])).all()


def test_shaping_is_monotone(data):
    """Order is preserved, so a stronger agreement never ends up smaller than a
    weaker one - otherwise the ranking the evidence rests on is destroyed."""
    linear = TradingAgent(AgentConfig()).signal(data)
    shaped = TradingAgent(AgentConfig(signal_shape=2.0)).signal(data)
    order_linear = linear.abs().rank()
    order_shaped = shaped.abs().rank()
    assert order_linear.corr(order_shaped) > 0.999


def test_deadband_zeroes_only_the_weak_region(data):
    linear = TradingAgent(AgentConfig()).signal(data)
    band = TradingAgent(AgentConfig(signal_deadband=0.25)).signal(data)
    weak = linear.abs() < 0.25
    assert (band[weak] == 0.0).all()
    pd.testing.assert_series_equal(band[~weak], linear[~weak], rtol=0, atol=0)


def test_deadband_reduces_time_in_market(data):
    linear = TradingAgent(AgentConfig()).signal(data)
    band = TradingAgent(AgentConfig(signal_deadband=0.30)).signal(data)
    assert (band.abs() > 1e-9).mean() <= (linear.abs() > 1e-9).mean()


def test_deadband_and_shape_compose_in_a_fixed_order(data):
    """Deadband first, then shape. Shaping first would push values below the
    band that were above it, making the band's meaning depend on the exponent."""
    both = TradingAgent(AgentConfig(signal_deadband=0.25, signal_shape=2.0)).signal(data)
    linear = TradingAgent(AgentConfig()).signal(data)
    weak = linear.abs() < 0.25
    assert (both[weak] == 0.0).all()
    strong = linear.abs() >= 0.25
    expected = np.sign(linear[strong]) * linear[strong].abs() ** 2.0
    np.testing.assert_allclose(both[strong].to_numpy(), expected.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("cfg", [
    {"signal_shape": 2.0},
    {"signal_deadband": 0.25},
    {"signal_deadband": 0.25, "signal_shape": 1.5},
])
def test_shaping_introduces_no_lookahead(data, cfg):
    """Future poisoning, same treatment as every other signal path."""
    cut = 600
    dirty = data.copy()
    rng = np.random.default_rng(0)
    shock = 500.0 * np.exp(np.cumsum(rng.normal(-0.01, 0.15, size=len(data) - cut)))
    for col, mult in (("close", 1.0), ("open", 1.01), ("high", 1.05), ("low", 0.95)):
        dirty.iloc[cut:, dirty.columns.get_loc(col)] = shock * mult
    a = TradingAgent(AgentConfig(**cfg)).signal(data)
    b = TradingAgent(AgentConfig(**cfg)).signal(dirty)
    pd.testing.assert_series_equal(a.iloc[:cut], b.iloc[:cut], rtol=0, atol=0)


def test_params_to_configs_passes_the_knobs_through():
    from tradingagent.engine import ExecutionConfig
    from tradingagent.optimize import DEFAULT_SEARCH_SPACE, params_to_configs

    params = {k: v[0] for k, v in DEFAULT_SEARCH_SPACE.items()}
    agent_cfg, _, _ = params_to_configs(params, ExecutionConfig())
    assert agent_cfg.signal_deadband == 0.0
    assert agent_cfg.signal_shape == 1.0

    params2 = {**params, "signal_deadband": 0.25, "signal_shape": 2.0}
    agent_cfg2, _, _ = params_to_configs(params2, ExecutionConfig())
    assert agent_cfg2.signal_deadband == 0.25
    assert agent_cfg2.signal_shape == 2.0


# --------------------------------------------------------------------------- #
# the trend floor
# --------------------------------------------------------------------------- #
def test_trend_floor_default_is_neutral(data):
    a = TradingAgent(AgentConfig()).signal(data)
    b = TradingAgent(AgentConfig(trend_floor=0.0)).signal(data)
    pd.testing.assert_series_equal(a, b, rtol=0, atol=0)


def test_trend_floor_only_lifts_never_lowers(data):
    """It is a floor. It may raise a weak long, never cut a strong one."""
    base = TradingAgent(AgentConfig()).signal(data)
    floored = TradingAgent(AgentConfig(trend_floor=0.4, trend_floor_lookback=200)).signal(data)
    assert (floored >= base - 1e-12).all()


def test_trend_floor_never_flips_a_short_into_a_long(data):
    """A floor that reversed a direction would be a different strategy."""
    cfg = AgentConfig(allow_short=True, trend_floor=0.5, trend_floor_lookback=200)
    base = TradingAgent(AgentConfig(allow_short=True)).signal(data)
    floored = TradingAgent(cfg).signal(data)
    shorts = base < 0
    pd.testing.assert_series_equal(floored[shorts], base[shorts], rtol=0, atol=0)


def test_trend_floor_does_not_apply_in_a_downtrend(data):
    """The defensive behaviour is the part worth keeping, so the floor must be
    absent exactly where the long-horizon trend is down."""
    lookback = 200
    down = data["close"].pct_change(lookback) <= 0
    base = TradingAgent(AgentConfig()).signal(data)
    floored = TradingAgent(AgentConfig(trend_floor=0.5, trend_floor_lookback=lookback)).signal(data)
    pd.testing.assert_series_equal(floored[down], base[down], rtol=0, atol=0)


def test_trend_floor_raises_time_in_market(data):
    base = TradingAgent(AgentConfig()).signal(data)
    floored = TradingAgent(AgentConfig(trend_floor=0.4, trend_floor_lookback=200)).signal(data)
    assert (floored.abs() > 1e-9).mean() >= (base.abs() > 1e-9).mean()


def test_trend_floor_introduces_no_lookahead(data):
    cut = 600
    dirty = data.copy()
    rng = np.random.default_rng(1)
    shock = 500.0 * np.exp(np.cumsum(rng.normal(-0.01, 0.15, size=len(data) - cut)))
    for col, mult in (("close", 1.0), ("open", 1.01), ("high", 1.05), ("low", 0.95)):
        dirty.iloc[cut:, dirty.columns.get_loc(col)] = shock * mult
    cfg = AgentConfig(trend_floor=0.4, trend_floor_lookback=200)
    a = TradingAgent(cfg).signal(data)
    b = TradingAgent(cfg).signal(dirty)
    pd.testing.assert_series_equal(a.iloc[:cut], b.iloc[:cut], rtol=0, atol=0)
