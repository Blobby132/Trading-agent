"""The market-structure screen.

It decides whether an asset is worth backtesting at all. Two things must hold:
the statistics must be causal and correctly computed, and the screen must be
honest about how weak a single estimate is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagent.data import synthetic_ohlcv
from tradingagent.market_structure import (
    FAVOURABLE,
    analyse,
    autocorrelation_stderr,
    compare,
    effective_observations,
    momentum_autocorrelation,
)


def _series(values) -> pd.Series:
    idx = pd.date_range("2016-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx)


def test_a_persistent_series_has_positive_momentum_autocorrelation():
    """Returns that persist must register as continuation.

    Built as an AR(1) on log returns with a positive coefficient, so successive
    stretches genuinely drift the same way. A pure exponential will NOT do: its
    period returns are constant, the variance is zero and the correlation is
    undefined rather than high - which is what a first version of this test got
    wrong.
    """
    rng = np.random.default_rng(0)
    n, phi = 4000, 0.97
    shocks = rng.normal(0, 0.01, n)
    lr = np.zeros(n)
    for i in range(1, n):
        lr[i] = phi * lr[i - 1] + shocks[i]
    persistent = _series(100 * np.exp(np.cumsum(lr)))
    assert momentum_autocorrelation(persistent, 63) > 0.1


def test_a_mean_reverting_series_has_negative_momentum_autocorrelation():
    """The sign is the whole finding, so both directions get a test."""
    t = np.arange(2000)
    oscillating = _series(100 * np.exp(0.25 * np.sin(2 * np.pi * t / 126.0)))
    assert momentum_autocorrelation(oscillating, 63) < 0.0


def test_a_constant_growth_series_gives_no_reading():
    """The degenerate case, pinned so nobody reintroduces it as a passing test:
    constant period returns have zero variance and no correlation to report."""
    t = np.arange(1500)
    exponential = _series(100 * np.exp(0.0008 * t))
    assert np.isnan(momentum_autocorrelation(exponential, 63))


def test_autocorrelation_is_nan_on_too_short_history():
    assert np.isnan(momentum_autocorrelation(_series(np.arange(50.0)), 63))


def test_standard_error_uses_independent_windows_not_bars():
    """The trap this function exists to avoid.

    A decade of daily bars looks like 3,650 observations and is really about 58
    independent 3-month windows. Using the bar count would understate the error
    roughly eight-fold and make a noisy reading look decisive.
    """
    n_bars, horizon = 3650, 63
    assert effective_observations(n_bars, horizon) == 56
    se = autocorrelation_stderr(n_bars, horizon)
    assert se == pytest.approx(1.0 / np.sqrt(56), rel=1e-9)
    naive = 1.0 / np.sqrt(n_bars)
    assert se > 7 * naive


def test_the_screen_is_a_screen_not_a_signal():
    """Nothing here may be usable as a trading decision: the module must expose
    no per-bar series, only whole-sample descriptive statistics."""
    import tradingagent.market_structure as ms

    data = synthetic_ohlcv(1200, seed=3)
    report = ms.analyse(data, symbol="X")
    assert all(np.isscalar(v) for v in report.stats.values())


def test_analyse_reports_all_four_conditions():
    data = synthetic_ohlcv(1500, seed=11)
    report = analyse(data, symbol="X")
    assert set(report.passes) == {"ac_1m_positive", "ac_3m_positive",
                                  "sideways_low", "bear_present"}
    assert 0 <= report.score <= 4
    assert report.verdict


def test_verdict_thresholds():
    data = synthetic_ohlcv(1200, seed=5)
    r = analyse(data, symbol="X")
    r.passes = {k: True for k in r.passes}
    assert r.score == 4 and "candidate" in r.verdict
    r.passes = {k: (i < 2) for i, k in enumerate(r.passes)}
    assert r.score == 2 and "marginal" in r.verdict
    r.passes = {k: False for k in r.passes}
    assert r.score == 0 and "unsuitable" in r.verdict


def test_thresholds_on_autocorrelation_are_zero_not_fitted():
    """A sign change is a statement about the market. A midpoint between six
    observed points would be a threshold fitted to six points."""
    assert FAVOURABLE["ac_1m_above"] == 0.0
    assert FAVOURABLE["ac_3m_above"] == 0.0


def test_compare_ranks_by_structural_match():
    frames = {f"S{i}": synthetic_ohlcv(1200, seed=20 + i) for i in range(4)}
    table = compare(frames)
    assert len(table) == 4
    assert list(table["score"]) == sorted(table["score"], reverse=True)
    assert {"symbol", "score", "verdict", "ac_3m", "sideways_share"} <= set(table.columns)


def test_screen_uses_only_trailing_information():
    """Future poisoning: the statistics describe a whole sample, but the regime
    labels inside them must still be causal, or the screen would quietly become
    a look-ahead device if anyone reused those labels."""
    from tradingagent.robustness import classify_regimes

    clean = synthetic_ohlcv(1400, seed=7)
    dirty = clean.copy()
    cut = 900
    rng = np.random.default_rng(0)
    shock = 500.0 * np.exp(np.cumsum(rng.normal(-0.01, 0.15, size=len(clean) - cut)))
    for col, mult in (("close", 1.0), ("open", 1.01), ("high", 1.05), ("low", 0.95)):
        dirty.iloc[cut:, dirty.columns.get_loc(col)] = shock * mult
    a = classify_regimes(clean["close"])["trend"].iloc[:cut]
    b = classify_regimes(dirty["close"])["trend"].iloc[:cut]
    pd.testing.assert_series_equal(a, b)
