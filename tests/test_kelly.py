"""The Kelly estimator has to be statistically honest or stay switched off."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagent.risk import RiskConfig, kelly_fraction


def series(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=idx)


def test_kelly_is_disabled_by_default():
    """The default configuration must not size on it."""
    assert RiskConfig().kelly_lookback == 0
    flat = kelly_fraction(series(np.random.default_rng(0).normal(0, 0.01, 500)), lookback=0)
    assert (flat == 1.0).all()


def test_no_position_before_the_sample_floor():
    """A short history produces zero, not a guess."""
    rng = np.random.default_rng(1)
    returns = series(rng.normal(0.002, 0.01, 300))
    out = kelly_fraction(returns, lookback=30, min_observations=120)
    assert (out.iloc[:120] == 0.0).all()
    assert (out.iloc[130:] > 0).any()


def test_pure_noise_produces_almost_no_leverage():
    """The failure this guards against: noise converted into position size."""
    rng = np.random.default_rng(2)
    noise = series(rng.normal(0.0, 0.01, 2000))       # zero true edge
    shrunk = kelly_fraction(noise, lookback=250, shrink=True)
    unshrunk = kelly_fraction(noise, lookback=250, shrink=False, cap=4.0)
    assert shrunk.mean() < unshrunk.mean()
    assert shrunk.quantile(0.95) < 0.5, "noise still produced meaningful sizing"


def test_a_real_edge_is_still_sized():
    """Shrinkage must not switch off a genuine, well-established edge."""
    rng = np.random.default_rng(3)
    strong = series(rng.normal(0.004, 0.01, 2000))    # Sharpe ~6 annualised
    out = kelly_fraction(strong, lookback=250)
    assert out.iloc[500:].mean() > 0.5


def test_shrinkage_scales_with_confidence():
    rng = np.random.default_rng(4)
    weak = series(rng.normal(0.0005, 0.01, 2000))
    strong = series(rng.normal(0.004, 0.01, 2000))
    assert kelly_fraction(weak, lookback=250).mean() < kelly_fraction(strong, lookback=250).mean()


def test_cap_defaults_to_full_kelly_not_beyond():
    """Above full Kelly, expected growth falls while variance keeps rising."""
    rng = np.random.default_rng(5)
    huge = series(rng.normal(0.05, 0.01, 1000))
    assert kelly_fraction(huge, lookback=250).max() <= 1.0
    assert kelly_fraction(huge, lookback=250, cap=2.5).max() <= 2.5


def test_kelly_never_goes_short():
    rng = np.random.default_rng(6)
    losing = series(rng.normal(-0.003, 0.01, 1000))
    assert (kelly_fraction(losing, lookback=250) >= 0).all()


def test_kelly_is_causal():
    """Today's size cannot use today's realised return."""
    rng = np.random.default_rng(7)
    returns = series(rng.normal(0.001, 0.01, 800))
    poisoned = returns.copy()
    poisoned.iloc[500:] = rng.normal(0.05, 0.01, 300)      # a different future
    clean = kelly_fraction(returns, lookback=250)
    dirty = kelly_fraction(poisoned, lookback=250)
    pd.testing.assert_series_equal(clean.iloc[:500], dirty.iloc[:500], rtol=0, atol=0)


def test_config_plumbs_the_guards_through():
    cfg = RiskConfig(kelly_lookback=250, kelly_min_observations=300, kelly_cap=0.5)
    assert cfg.kelly_min_observations == 300 and cfg.kelly_cap == 0.5
