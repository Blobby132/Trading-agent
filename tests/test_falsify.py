"""The falsification suite, and the properties that make it a valid null.

A null experiment is only worth running if it removes the thing it claims to
remove and preserves everything else. These tests pin both halves, because a
null that quietly changes turnover or exposure produces a comparison that looks
rigorous and is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagent.data import synthetic_ohlcv
from tradingagent.engine import ExecutionConfig
from tradingagent.falsify import (
    block_permute,
    bootstrap_ohlcv,
    circular_block_bootstrap,
    compare_to_null,
    flip_signs,
    null_suite,
    shuffle_weights,
)
from tradingagent.optimize import WalkForwardConfig, walk_forward


# --------------------------------------------------------------------------- #
# resampling primitives
# --------------------------------------------------------------------------- #
def test_block_permute_is_a_permutation():
    """Every value survives - a null that drops data is not a null."""
    v = np.arange(100.0)
    out = block_permute(v, 10, np.random.default_rng(0))
    assert len(out) == len(v)
    np.testing.assert_array_equal(np.sort(out), np.sort(v))


def test_block_permute_preserves_within_block_order():
    v = np.arange(100.0)
    out = block_permute(v, 10, np.random.default_rng(3))
    # each length-10 run in the output must be 10 consecutive originals
    for start in range(0, 100, 10):
        chunk = out[start : start + 10]
        np.testing.assert_array_equal(chunk, np.arange(chunk[0], chunk[0] + len(chunk)))


def test_block_permute_actually_reorders():
    v = np.arange(200.0)
    out = block_permute(v, 10, np.random.default_rng(1))
    assert not np.array_equal(out, v), "permutation left the series untouched"


def test_circular_bootstrap_length_and_membership():
    v = np.arange(50.0)
    out = circular_block_bootstrap(v, 7, np.random.default_rng(0))
    assert len(out) == len(v)
    assert set(np.unique(out)) <= set(v)


def test_circular_bootstrap_samples_the_tail():
    """A plain block bootstrap under-samples the ends; circular must not.

    For a price series that would mean under-sampling whichever era sits last,
    which is a silent bias in the null itself.
    """
    v = np.arange(100.0)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        seen.update(np.unique(circular_block_bootstrap(v, 10, rng)).tolist())
    assert 99.0 in seen and 0.0 in seen


# --------------------------------------------------------------------------- #
# the synthetic price path
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real() -> pd.DataFrame:
    return synthetic_ohlcv(800, seed=4)


def test_bootstrapped_prices_keep_the_ohlc_bracket(real):
    fake = bootstrap_ohlcv(real, block=21, rng=np.random.default_rng(0))
    assert (fake["high"] >= fake[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (fake["low"] <= fake[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (fake["close"] > 0).all()
    assert len(fake) == len(real)
    pd.testing.assert_index_equal(fake.index, real.index)


def test_bootstrapped_prices_keep_short_horizon_character(real):
    """Volatility must survive; only the long-horizon path should be destroyed.

    If the bootstrap also flattened volatility, the null would be easier than
    reality and the strategy would beat it for the wrong reason.
    """
    fake = bootstrap_ohlcv(real, block=21, rng=np.random.default_rng(0))
    r_real = np.log(real["close"]).diff().dropna()
    r_fake = np.log(fake["close"]).diff().dropna()
    assert r_fake.std() == pytest.approx(r_real.std(), rel=0.20)


def test_bootstrapped_prices_destroy_long_horizon_trend(real):
    """The thing a trend model claims to exploit has to actually be gone.

    Averaged over many draws the 63-bar autocorrelation of the bootstrapped
    path must sit closer to zero than the real path's.
    """
    real_ac = abs(np.log(real["close"]).diff().dropna().autocorr(lag=63))
    fakes = [
        abs(np.log(bootstrap_ohlcv(real, block=21, rng=np.random.default_rng(s))["close"])
            .diff().dropna().autocorr(lag=63))
        for s in range(12)
    ]
    assert np.nanmedian(fakes) <= max(real_ac, 0.05) + 0.02


def test_bootstrapped_prices_differ_between_seeds(real):
    a = bootstrap_ohlcv(real, block=21, rng=np.random.default_rng(1))
    b = bootstrap_ohlcv(real, block=21, rng=np.random.default_rng(2))
    assert not np.allclose(a["close"].to_numpy(), b["close"].to_numpy())


# --------------------------------------------------------------------------- #
# weight-level nulls preserve what they must
# --------------------------------------------------------------------------- #
def test_shuffled_weights_preserve_the_exposure_profile():
    """Same weights, same time in market - only the days change.

    This is what makes it a fair null: cost and exposure are held constant, so
    any difference in outcome is attributable to timing alone.
    """
    idx = pd.date_range("2020-01-01", periods=300, tz="UTC")
    w = pd.Series(np.sin(np.arange(300) / 10.0), index=idx)
    out = shuffle_weights(21, 5)(w)
    np.testing.assert_allclose(np.sort(out.to_numpy()), np.sort(w.to_numpy()))
    assert out.abs().mean() == pytest.approx(w.abs().mean())


def test_sign_flipping_preserves_magnitude():
    idx = pd.date_range("2020-01-01", periods=200, tz="UTC")
    w = pd.Series(np.linspace(0.1, 1.0, 200), index=idx)
    out = flip_signs(21, 7)(w)
    np.testing.assert_allclose(out.abs().to_numpy(), w.abs().to_numpy())
    assert (out.to_numpy() < 0).any(), "no sign was actually flipped"


def test_weight_nulls_are_deterministic_for_a_seed():
    idx = pd.date_range("2020-01-01", periods=120, tz="UTC")
    w = pd.Series(np.arange(120.0), index=idx)
    pd.testing.assert_series_equal(shuffle_weights(10, 3)(w), shuffle_weights(10, 3)(w))
    pd.testing.assert_series_equal(flip_signs(10, 3)(w), flip_signs(10, 3)(w))


# --------------------------------------------------------------------------- #
# the random-selection null
# --------------------------------------------------------------------------- #
def test_random_selection_differs_from_best_selection():
    """The null must actually cripple the optimiser, not quietly agree with it."""
    data = synthetic_ohlcv(1200, seed=8)
    cfg = WalkForwardConfig(train_bars=400, test_bars=150, n_candidates=12, top_k=3,
                            seed=0, verbose=False)
    best = walk_forward(data, ExecutionConfig(initial_capital=100.0), cfg, selection="best")
    rand = walk_forward(data, ExecutionConfig(initial_capital=100.0), cfg, selection="random")
    assert best.chosen != rand.chosen, "random selection picked the same configurations as best"
    # the candidate pool and the training work are identical - only the pick changed
    assert best.n_evaluations == rand.n_evaluations


def test_selection_must_be_a_known_mode():
    data = synthetic_ohlcv(700, seed=2)
    cfg = WalkForwardConfig(train_bars=300, test_bars=120, n_candidates=4, top_k=2, verbose=False)
    with pytest.raises(ValueError, match="selection must be"):
        walk_forward(data, ExecutionConfig(), cfg, selection="whatever")


def test_random_selection_is_reproducible():
    data = synthetic_ohlcv(1000, seed=9)
    cfg = WalkForwardConfig(train_bars=350, test_bars=150, n_candidates=10, top_k=2,
                            seed=4, verbose=False)
    a = walk_forward(data, ExecutionConfig(), cfg, selection="random")
    b = walk_forward(data, ExecutionConfig(), cfg, selection="random")
    pd.testing.assert_series_equal(a.equity, b.equity)


# --------------------------------------------------------------------------- #
# the suite and its comparison
# --------------------------------------------------------------------------- #
def test_null_suite_runs_every_null_and_returns_rows():
    data = synthetic_ohlcv(900, seed=11)
    cfg = WalkForwardConfig(train_bars=350, test_bars=150, n_candidates=4, top_k=2,
                            seed=0, verbose=False)
    out = null_suite(data, base_exec=ExecutionConfig(initial_capital=100.0), wf=cfg,
                     replications=2, block=21, verbose=False)
    assert set(out["null"]) == {
        "shuffled_signal", "sign_flipped", "bootstrapped_prices", "random_selection"
    }
    assert len(out) == 8
    assert out["sharpe"].notna().any()


def test_compare_to_null_p_value_bounds():
    """With R replications the smallest reportable p-value is 1/(R+1).

    Quoting anything smaller would be an artefact of the replication count.
    """
    nulls = pd.DataFrame({"null": ["x"] * 20, "sharpe": np.linspace(-1, 1, 20)})
    beats_all = compare_to_null({"sharpe": 99.0}, nulls)
    assert beats_all["p_value"].iloc[0] == pytest.approx(1 / 21)
    loses_to_all = compare_to_null({"sharpe": -99.0}, nulls)
    assert loses_to_all["p_value"].iloc[0] == pytest.approx(1.0)


def test_compare_to_null_reports_the_percentile():
    nulls = pd.DataFrame({"null": ["x"] * 100, "sharpe": np.linspace(0, 1, 100)})
    out = compare_to_null({"sharpe": 0.5}, nulls)
    assert out["percentile_of_observed"].iloc[0] == pytest.approx(50.0, abs=2.0)
    assert out["null_median"].iloc[0] == pytest.approx(0.5, abs=0.02)
