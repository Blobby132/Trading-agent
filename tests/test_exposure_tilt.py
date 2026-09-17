"""Exposure tilt: redistribute exposure across trend states, never add any.

The tilt exists to move capital from weak, over-extended states into strong,
steady ones. Its defining safety property is that it is a *redistribution*: the
ranks it keys off are uniform by construction, so the average multiplier is one.
These tests pin that, the causality, and the ordering against the protected
trend floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagent.agent import AgentConfig, TradingAgent
from tradingagent.data import synthetic_ohlcv


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return synthetic_ohlcv(1500, seed=23)


def test_defaults_are_exactly_neutral(data):
    a = TradingAgent(AgentConfig()).signal(data)
    b = TradingAgent(AgentConfig(trend_tilt=0.0, accel_tilt=0.0)).signal(data)
    pd.testing.assert_series_equal(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("cfg", [
    {"trend_tilt": 0.25}, {"trend_tilt": 0.5},
    {"accel_tilt": 0.25}, {"accel_tilt": 0.5},
    {"trend_tilt": 0.25, "accel_tilt": 0.25},
])
def test_the_tilt_is_approximately_exposure_neutral(data, cfg):
    """The whole justification. If the tilt raised average exposure it would be
    a leverage increase wearing a signal's clothes, and any return improvement
    would be uninterpretable."""
    base = TradingAgent(AgentConfig()).signal(data).abs().mean()
    tilted = TradingAgent(AgentConfig(**cfg)).signal(data).abs().mean()
    assert tilted == pytest.approx(base, rel=0.15), (base, tilted)


@pytest.mark.parametrize("k", [0.25, 0.5, 1.0])
def test_the_tilt_cannot_breach_the_exposure_cap(data, k):
    t = TradingAgent(AgentConfig(trend_tilt=k)).signal(data)
    assert t.abs().max() <= 1.0 + 1e-12


def test_the_tilt_never_makes_a_long_negative(data):
    """A multiplier floored at zero can shrink a position to nothing but must
    never invert it - that would be a different strategy, not a tilt."""
    base = TradingAgent(AgentConfig()).signal(data)
    for cfg in ({"trend_tilt": 1.0}, {"accel_tilt": 1.0}, {"trend_tilt": 1.0, "accel_tilt": 1.0}):
        t = TradingAgent(AgentConfig(**cfg)).signal(data)
        longs = base > 0
        assert (t[longs] >= 0).all(), cfg


def test_trend_tilt_raises_exposure_in_strong_trends_and_lowers_it_in_weak(data):
    """Monotone and in the direction the screen justified."""
    base = TradingAgent(AgentConfig()).signal(data)
    tilted = TradingAgent(AgentConfig(trend_tilt=0.5)).signal(data)
    strength = data["close"].pct_change(90)
    rank = strength.rolling(365, min_periods=90).rank(pct=True)
    live = (base.abs() > 1e-6) & rank.notna()
    ratio = (tilted[live] / base[live])
    top, bottom = rank[live] > 0.75, rank[live] < 0.25
    assert ratio[top].mean() > 1.0
    assert ratio[bottom].mean() < 1.0


def test_accel_tilt_points_the_other_way(data):
    """The screen found high acceleration predicts LOWER forward returns, so the
    tilt must reduce exposure there. Getting this sign backwards would be an
    easy and invisible mistake."""
    base = TradingAgent(AgentConfig()).signal(data)
    tilted = TradingAgent(AgentConfig(accel_tilt=0.5)).signal(data)
    accel = data["close"].pct_change(90) - data["close"].pct_change(180)
    rank = accel.rolling(365, min_periods=90).rank(pct=True)
    live = (base.abs() > 1e-6) & rank.notna()
    ratio = tilted[live] / base[live]
    assert ratio[rank[live] > 0.75].mean() < 1.0
    assert ratio[rank[live] < 0.25].mean() > 1.0


def test_the_floor_still_binds_under_a_tilt(data):
    """The floor is a protected component. The tilt is applied before it, so a
    tilt that shrinks a position cannot push it below the guaranteed minimum."""
    floor = 0.5
    cfg = AgentConfig(trend_floor=floor, trend_floor_lookback=200,
                      trend_tilt=0.5, accel_tilt=0.5)
    out = TradingAgent(cfg).signal(data)
    up = data["close"].pct_change(200) > 0
    live = up.reindex(out.index).fillna(False) & (out >= 0)
    assert (out[live] >= floor - 1e-12).all()


def test_warmup_is_no_opinion_not_go_flat(data):
    """Before the rank window fills there is no rank. Treating that as a zero
    multiplier would silently flatten the book through every warm-up."""
    base = TradingAgent(AgentConfig()).signal(data)
    tilted = TradingAgent(AgentConfig(trend_tilt=0.5, tilt_rank_window=900)).signal(data)
    early = slice(0, 100)
    pd.testing.assert_series_equal(tilted.iloc[early], base.iloc[early], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("cfg", [
    {"trend_tilt": 0.5}, {"accel_tilt": 0.5}, {"trend_tilt": 0.25, "accel_tilt": 0.25},
])
def test_tilt_introduces_no_lookahead(data, cfg):
    cut = 900
    dirty = data.copy()
    rng = np.random.default_rng(2)
    shock = 500.0 * np.exp(np.cumsum(rng.normal(-0.01, 0.15, size=len(data) - cut)))
    for col, mult in (("close", 1.0), ("open", 1.01), ("high", 1.05), ("low", 0.95)):
        dirty.iloc[cut:, dirty.columns.get_loc(col)] = shock * mult
    a = TradingAgent(AgentConfig(**cfg)).signal(data)
    b = TradingAgent(AgentConfig(**cfg)).signal(dirty)
    pd.testing.assert_series_equal(a.iloc[:cut], b.iloc[:cut], rtol=0, atol=0)


def test_tilt_knobs_are_classified_for_rescaling():
    from tradingagent.timescale import BAR_COUNT_PARAMS, SCALE_INVARIANT_PARAMS
    for k in ("tilt_lookback", "tilt_rank_window"):
        assert k in BAR_COUNT_PARAMS["AgentConfig"], k
    for k in ("trend_tilt", "accel_tilt"):
        assert k in SCALE_INVARIANT_PARAMS, k
