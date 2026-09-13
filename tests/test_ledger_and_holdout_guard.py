"""The research process has to be auditable, and the holdout has to be hard to spend.

Walk-forward keeps the optimiser out of the window it trades. Nothing in the
code could previously keep the *researcher* out of the holdout, and nothing
recorded how many times a search had been run. These tests pin both.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from tradingagent.data import synthetic_ohlcv
from tradingagent.engine import ExecutionConfig
from tradingagent.holdout import Holdout, HoldoutViolation
from tradingagent.ledger import ResearchLedger
from tradingagent.optimize import WalkForwardConfig, walk_forward
from tradingagent.universe import Panel
from tradingagent.xs_optimize import XSWalkForwardConfig, walk_forward_xs

SMALL = WalkForwardConfig(train_bars=300, test_bars=150, n_candidates=3, top_k=2,
                          seed=0, verbose=False)


@pytest.fixture
def prices():
    return synthetic_ohlcv(1200, seed=3, start="2018-01-01")


@pytest.fixture
def ledger(tmp_path):
    return ResearchLedger(str(tmp_path / "ledger.jsonl"))


# --------------------------------------------------------------------------- #
# holdout protection
# --------------------------------------------------------------------------- #
def test_optimizer_refuses_data_reaching_into_the_holdout(prices):
    holdout = Holdout("2020-06-01", name="guard_test")
    with pytest.raises(HoldoutViolation, match="reserved era"):
        walk_forward(prices, ExecutionConfig(), SMALL, holdout=holdout)


def test_optimizer_accepts_the_development_split(prices):
    holdout = Holdout("2020-06-01", name="guard_test")
    result = walk_forward(holdout.development(prices), ExecutionConfig(), SMALL,
                          holdout=holdout)
    assert result.equity.index[-1] < holdout.start


def test_cross_sectional_optimizer_is_guarded_too():
    panel = Panel.from_frames(
        {f"S{i}": synthetic_ohlcv(1200, seed=400 + i, start="2018-01-01") for i in range(15)}
    )
    holdout = Holdout("2020-06-01", name="xs_guard")
    cfg = XSWalkForwardConfig(train_bars=300, test_bars=150, embargo_bars=5,
                              n_candidates=2, top_k=1, verbose=False)
    with pytest.raises(HoldoutViolation):
        walk_forward_xs(panel, ExecutionConfig(periods_per_year=252.0), cfg, holdout=holdout)


def test_guard_is_a_no_op_when_data_stops_before_the_holdout(prices):
    holdout = Holdout("2030-01-01", name="far_future")
    assert holdout.guard(prices) is prices


def test_protect_trims_instead_of_raising(prices):
    holdout = Holdout("2020-06-01", name="protect_test")
    trimmed = holdout.protect(prices)
    assert trimmed.index[-1] < holdout.start
    assert len(trimmed) < len(prices)


def test_guard_message_names_the_escape_hatches(prices):
    holdout = Holdout("2020-06-01", name="msg_test")
    with pytest.raises(HoldoutViolation) as excinfo:
        holdout.guard(prices, what="my search")
    message = str(excinfo.value)
    assert "my search" in message
    assert "holdout.development" in message and "holdout.evaluate" in message


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #
def test_walk_forward_records_itself(prices, ledger):
    walk_forward(prices, ExecutionConfig(), SMALL, ledger=ledger, label="run one")
    rows = ledger.entries()
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "run one"
    assert row["used_for_selection"] is True
    assert row["candidate_count"] > 0
    assert row["seed"] == SMALL.seed
    assert "sharpe" in row["stats"] and "max_drawdown" in row["stats"]
    assert row["experiment_id"] and row["timestamp"]


def test_effective_trials_accumulate_across_runs(prices, ledger):
    """Searches you ran last week still happened."""
    for seed in (1, 2, 3):
        walk_forward(prices, ExecutionConfig(),
                     WalkForwardConfig(**{**SMALL.__dict__, "seed": seed}), ledger=ledger)
    total = ledger.selection_count()
    assert total == sum(r["candidate_count"] for r in ledger.entries())
    assert total > SMALL.n_candidates, "three searches counted as one"


def test_non_selecting_runs_do_not_inflate_the_trial_count(ledger):
    ledger.record("a search", {"sharpe": 1.0}, candidate_count=100, used_for_selection=True)
    ledger.record("just looking", {"sharpe": 2.0}, candidate_count=100,
                  used_for_selection=False)
    assert ledger.selection_count() == 100


def test_ledger_survives_a_new_object_and_is_append_only(ledger):
    ledger.record("first", {"sharpe": 1.0})
    reopened = ResearchLedger(ledger.path)
    reopened.record("second", {"sharpe": 2.0})
    labels = [r["label"] for r in ResearchLedger(ledger.path).entries()]
    assert labels == ["first", "second"]


def test_ledger_tolerates_a_corrupt_line(ledger):
    ledger.record("good", {"sharpe": 1.0})
    with open(ledger.path, "a") as fh:
        fh.write("{not json at all\n")
    ledger.record("also good", {"sharpe": 1.5})
    assert [r["label"] for r in ledger.entries()] == ["good", "also good"]


def test_ledger_stores_only_finite_tracked_stats(ledger):
    ledger.record("weird", {"sharpe": float("nan"), "calmar": float("inf"),
                            "final_equity": 250.0, "not_tracked": 7.0})
    stats = ledger.entries()[0]["stats"]
    assert stats == {"final_equity": 250.0}


def test_ledger_serialises_numpy_parameters(ledger):
    ledger.record("numpy", {"sharpe": 1.0},
                  parameters={"lookback": np.int64(50), "vol": np.float64(0.2),
                              "on": np.bool_(True), "when": pd.Timestamp("2024-01-01")})
    params = ledger.entries()[0]["parameters"]
    assert params["lookback"] == 50 and params["vol"] == 0.2 and params["on"] is True
    assert isinstance(params["when"], str)


def test_summary_warns_once_the_trial_count_is_large(ledger):
    ledger.record("big search", {"sharpe": 1.0}, candidate_count=5000,
                  used_for_selection=True)
    assert "deflated" in ledger.summary()


def test_frame_flattens_stats_for_analysis(ledger):
    ledger.record("one", {"sharpe": 1.2, "final_equity": 300.0}, universe="us")
    ledger.record("two", {"sharpe": 0.8, "final_equity": 200.0}, universe="us")
    frame = ledger.frame()
    assert list(frame["label"]) == ["one", "two"]
    assert frame["sharpe"].tolist() == [1.2, 0.8]
