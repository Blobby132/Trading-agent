import numpy as np
import pandas as pd
import pytest

from tradingagent.data import align_universe, synthetic_ohlcv
from tradingagent.engine import ExecutionConfig
from tradingagent.optimize import (
    DEFAULT_SEARCH_SPACE,
    OBJECTIVES,
    WalkForwardConfig,
    params_to_configs,
    sample_unique,
    walk_forward,
)

SMALL = WalkForwardConfig(
    train_bars=300, test_bars=120, n_candidates=6, top_k=2, verbose=False, seed=0
)


@pytest.fixture(scope="module")
def long_prices():
    return synthetic_ohlcv(1200, seed=17)


def test_sampled_configurations_are_distinct():
    picks = sample_unique(25, DEFAULT_SEARCH_SPACE, seed=3)
    keys = {tuple(sorted(p.items())) for p in picks}
    assert len(keys) == 25


def test_small_space_is_enumerated_exhaustively():
    space = {"a": [1, 2], "b": [3, 4]}
    assert len(sample_unique(100, space, seed=0)) == 4


def test_params_build_valid_configs():
    params = sample_unique(1, DEFAULT_SEARCH_SPACE, seed=11)[0]
    agent_cfg, risk_cfg, exec_cfg = params_to_configs(params, ExecutionConfig())
    assert agent_cfg.strategies
    assert risk_cfg.max_leverage == exec_cfg.max_leverage


def test_walk_forward_only_reports_out_of_sample_bars(long_prices):
    res = walk_forward(long_prices, ExecutionConfig(), SMALL)
    # nothing before the first training window may appear in the traded curve
    first_test = res.folds["test_start"].min()
    assert res.equity.index.min() == first_test
    assert res.equity.index.max() <= long_prices.index.max()


def test_walk_forward_windows_do_not_overlap_their_training_data(long_prices):
    res = walk_forward(long_prices, ExecutionConfig(), SMALL)
    for _, row in res.folds.iterrows():
        assert row["train_end"] < row["test_start"], "training window leaked into the test window"


def test_walk_forward_equity_compounds_across_folds(long_prices):
    res = walk_forward(long_prices, ExecutionConfig(initial_capital=100.0), SMALL)
    folds = res.folds
    assert folds["start_equity"].iloc[0] == pytest.approx(100.0)
    for i in range(1, len(folds)):
        assert folds["start_equity"].iloc[i] == pytest.approx(folds["end_equity"].iloc[i - 1], rel=1e-9)


def test_walk_forward_rejects_too_little_data():
    tiny = synthetic_ohlcv(200, seed=1)
    with pytest.raises(ValueError, match="not enough data"):
        walk_forward(tiny, ExecutionConfig(), SMALL)


def test_walk_forward_handles_a_portfolio():
    panel = align_universe(
        {"A": synthetic_ohlcv(1200, seed=2), "B": synthetic_ohlcv(1200, seed=3)}
    )
    res = walk_forward(panel, ExecutionConfig(), SMALL)
    assert list(res.weights.columns) == ["A", "B"]


@pytest.mark.parametrize("name", sorted(OBJECTIVES))
def test_objectives_return_finite_scores(name):
    stats = {
        "sharpe": 1.1, "cagr": 0.5, "max_drawdown": -0.3, "final_equity": 250.0,
        "initial_equity": 100.0, "n_trades": 50, "bust": 0.0,
    }
    assert np.isfinite(OBJECTIVES[name](stats))


def test_target_growth_objective_rejects_a_bust():
    busted = {"final_equity": 0.0, "initial_equity": 100.0, "max_drawdown": -1.0,
              "n_trades": 5, "bust": 1.0}
    healthy = {"final_equity": 250.0, "initial_equity": 100.0, "max_drawdown": -0.2,
               "n_trades": 50, "bust": 0.0}
    assert OBJECTIVES["target_growth"](busted) < OBJECTIVES["target_growth"](healthy)


def test_target_growth_prefers_the_shallower_drawdown():
    calm = {"final_equity": 300.0, "initial_equity": 100.0, "max_drawdown": -0.20,
            "n_trades": 60, "bust": 0.0}
    wild = {"final_equity": 300.0, "initial_equity": 100.0, "max_drawdown": -0.75,
            "n_trades": 60, "bust": 0.0}
    assert OBJECTIVES["target_growth"](calm) > OBJECTIVES["target_growth"](wild)
