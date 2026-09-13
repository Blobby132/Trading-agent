import numpy as np
import pandas as pd
import pytest

from tradingagent.data import synthetic_ohlcv
from tradingagent.engine import ExecutionConfig
from tradingagent.universe import Panel
from tradingagent.xs_optimize import (
    XS_SEARCH_SPACE,
    XS_SEARCH_SPACE_LONG_ONLY,
    XSWalkForwardConfig,
    equal_weight_benchmark,
    params_to_rules,
    walk_forward_xs,
)

SMALL = XSWalkForwardConfig(
    train_bars=500, test_bars=150, embargo_bars=10, n_candidates=4, top_k=2,
    seed=0, verbose=False,
)


@pytest.fixture(scope="module")
def panel():
    frames = {f"S{i}": synthetic_ohlcv(1200, seed=100 + i) for i in range(20)}
    return Panel.from_frames(frames)


def test_walk_forward_runs_and_reports_only_traded_windows(panel):
    res = walk_forward_xs(panel, ExecutionConfig(periods_per_year=252.0), SMALL)
    assert res.equity.index.min() == res.folds["test_start"].min()
    assert len(res.folds) >= 2
    assert set(res.weights.columns) == set(panel.symbols)


def test_training_window_never_touches_the_test_window(panel):
    res = walk_forward_xs(panel, ExecutionConfig(periods_per_year=252.0), SMALL)
    for _, row in res.folds.iterrows():
        assert row["train_end"] < row["test_start"]
        gap = (row["test_start"] - row["train_end"]).days
        assert gap >= SMALL.embargo_bars, "embargo shorter than configured"


def test_equity_compounds_across_folds(panel):
    res = walk_forward_xs(
        panel, ExecutionConfig(initial_capital=100.0, periods_per_year=252.0), SMALL
    )
    folds = res.folds
    assert folds["start_equity"].iloc[0] == pytest.approx(100.0)
    for i in range(1, len(folds)):
        assert folds["start_equity"].iloc[i] == pytest.approx(
            folds["end_equity"].iloc[i - 1], rel=1e-9
        )


def test_long_only_space_never_shorts(panel):
    res = walk_forward_xs(
        panel, ExecutionConfig(periods_per_year=252.0), SMALL, XS_SEARCH_SPACE_LONG_ONLY
    )
    assert (res.weights >= -1e-9).all().all()
    assert res.folds["long_only"].all()


def test_coefficient_stability_reports_are_finite(panel):
    res = walk_forward_xs(panel, ExecutionConfig(periods_per_year=252.0), SMALL)
    stability = res.coefficient_stability()
    assert stability["n_folds"] >= 2
    for key, value in stability.items():
        assert not isinstance(value, str)
        assert np.isfinite(value) or key.endswith("correlation")


def test_walk_forward_rejects_too_little_history():
    tiny = Panel.from_frames({f"S{i}": synthetic_ohlcv(200, seed=i) for i in range(15)})
    with pytest.raises(ValueError, match="not enough data"):
        walk_forward_xs(tiny, ExecutionConfig(), SMALL)


def test_params_to_rules_maps_every_knob():
    params = {
        "long_only": 0, "top_frac": 0.25, "gross": 1.5, "max_weight": 0.2,
        "rebalance_every": 5, "weighting": "score",
    }
    rules = params_to_rules(params)
    assert rules.long_only is False
    assert rules.top_frac == rules.bottom_frac == 0.25
    assert rules.gross == 1.5 and rules.max_weight == 0.2
    assert rules.weighting == "score"


def test_equal_weight_benchmark_holds_the_whole_universe(panel):
    eq = equal_weight_benchmark(panel, 100.0)
    assert len(eq) == len(panel)
    assert eq.iloc[0] == pytest.approx(100.0, rel=0.05)
    assert np.isfinite(eq).all()


def test_search_space_keys_match_what_the_loop_reads():
    needed = {"ranker", "ridge_alpha", "horizon", "long_only", "top_frac",
              "rebalance_every", "max_weight", "gross", "weighting", "vol_target"}
    assert needed.issubset(XS_SEARCH_SPACE)
    assert needed.issubset(XS_SEARCH_SPACE_LONG_ONLY)


def test_reported_weights_are_the_realised_book_not_the_request(panel):
    """Exposure stats must describe positions, not intentions."""
    ec = ExecutionConfig(initial_capital=100.0, periods_per_year=252.0, max_leverage=1.0)
    res = walk_forward_xs(panel, ec, SMALL)
    gross = res.weights.abs().sum(axis=1)
    # intraday drift can nudge it slightly past the cap; a stored *request*
    # would sit far above it, because candidates may ask for 1.5x and get blended
    assert gross.max() <= 1.35, f"reported gross {gross.max():.2f} looks like a request"


def test_recorded_coefficients_are_unit_normalised(panel):
    from tradingagent.xs_optimize import _unit_norm

    assert _unit_norm({}) == {}
    assert _unit_norm({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}
    normed = _unit_norm({"a": 3.0, "b": 4.0})
    assert normed["a"] == pytest.approx(0.6)
    assert normed["b"] == pytest.approx(0.8)

    res = walk_forward_xs(panel, ExecutionConfig(periods_per_year=252.0), SMALL)
    norms = (res.coefficients.fillna(0.0) ** 2).sum(axis=1) ** 0.5
    assert ((norms - 1.0).abs() < 1e-9).all()
