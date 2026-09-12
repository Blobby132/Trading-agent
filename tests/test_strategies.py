import numpy as np
import pandas as pd
import pytest

from tradingagent.strategies import STRATEGY_REGISTRY, build_ensemble, make_strategy

NAMES = sorted(STRATEGY_REGISTRY)


@pytest.mark.parametrize("name", NAMES)
def test_signal_is_bounded_and_aligned(prices, name):
    w = make_strategy(name).target_weight(prices)
    assert len(w) == len(prices)
    assert w.index.equals(prices.index)
    assert w.between(-1.0, 1.0).all()
    assert not w.isna().any()


@pytest.mark.parametrize("name", NAMES)
def test_signal_is_causal(prices, name):
    """Hiding the future must not change a single past signal."""
    cut = 600
    strat = make_strategy(name)
    full = strat.target_weight(prices).iloc[:cut]
    truncated = strat.target_weight(prices.iloc[:cut])
    pd.testing.assert_series_equal(full, truncated, check_names=False, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("name", NAMES)
def test_long_only_mode_never_shorts(prices, name):
    space = STRATEGY_REGISTRY[name].param_space()
    if "allow_short" not in space:
        pytest.skip(f"{name} has no short mode")
    w = make_strategy(name, allow_short=0).target_weight(prices)
    assert (w >= 0).all()


def test_trend_strategies_go_long_in_an_uptrend(trending_prices):
    for name in ("ema_trend", "donchian", "ts_momentum"):
        w = make_strategy(name).target_weight(trending_prices)
        assert w.tail(100).mean() > 0.3, f"{name} failed to follow a clean uptrend"


def test_unknown_parameter_is_rejected():
    with pytest.raises(TypeError):
        make_strategy("ema_trend", not_a_real_param=1)


def test_unknown_strategy_is_rejected():
    with pytest.raises(KeyError):
        make_strategy("no_such_strategy")


def test_ensemble_builds_requested_members():
    ens = build_ensemble(["ema_trend", "donchian"], {"ema_trend": {"fast": 5, "slow": 40}})
    assert [s.name for s in ens] == ["ema_trend", "donchian"]
    assert ens[0].params["fast"] == 5


def test_inverted_ema_parameters_produce_no_signal(prices):
    """fast >= slow is meaningless; it must be inert, not silently reversed."""
    w = make_strategy("ema_trend", fast=100, slow=20).target_weight(prices)
    assert (w == 0).all()
