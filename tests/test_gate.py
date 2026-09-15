"""The acceptance gate. Default-deny, and silence counts as failure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagent.gate import DEFAULT_FLOORS, Check, GateResult, evaluate


def _everything_passing(**overrides):
    kw = dict(
        oos_stats={"sharpe": 1.2, "max_drawdown": -0.30, "n_trades": 200, "sortino": 1.5,
                   "final_equity": 2000.0},
        matched_baseline_stats={"sharpe": 0.6, "sortino": 0.8, "final_equity": 800.0},
        stability=pd.DataFrame({"parameter": ["a", "b"], "stability": [0.9, 0.95],
                                "verdict": ["plateau", "plateau"]}),
        cost_ladder=pd.DataFrame({"bps_per_side": [0, 15, 35, 75],
                                  "final_equity": [3000.0, 2500.0, 2000.0, 1200.0]}),
        regime_report=pd.DataFrame({"dimension": ["trend"] * 3,
                                    "regime": ["bull", "bear", "sideways"],
                                    "excess": [0.2, 0.3, -0.1]}),
        cross_asset=pd.DataFrame({"tier": ["unseen"] * 4,
                                  "beats_buy_hold": [True, True, True, False]}),
        deflated_sharpe=0.9,
        null_comparison=pd.DataFrame({"null": ["a", "b"], "p_value": [0.02, 0.05]}),
        capacity={"max_participation": 1e-4},
        parity_tests_pass=True,
        holdout_untouched=True,
    )
    kw.update(overrides)
    return evaluate(**kw)


# --------------------------------------------------------------------------- #
# default-deny
# --------------------------------------------------------------------------- #
def test_nothing_supplied_is_rejected():
    """The way a gate gets subverted is by not running the test, so an
    un-run check must never read as a pass."""
    r = evaluate()
    assert not r.accepted
    assert all(c.passed is None for c in r.checks)
    assert len(r.blockers) == 9          # every required check, none of the advisory one


def test_a_not_run_check_blocks_even_when_everything_else_passes():
    r = _everything_passing(cross_asset=None)
    assert not r.accepted
    assert [c.name for c in r.blockers] == ["cross_asset"]
    assert r.frame().set_index("check").loc["cross_asset", "status"] == "NOT RUN"


def test_a_complete_strong_result_is_accepted():
    """The gate must be passable, or it is not a gate."""
    r = _everything_passing()
    assert r.accepted, r.report()
    assert not r.blockers


def test_the_advisory_check_never_blocks():
    r = _everything_passing(holdout_untouched=False)
    assert r.accepted
    assert any(c.name == "holdout_discipline" and c.passed is False for c in r.checks)


# --------------------------------------------------------------------------- #
# individual dimensions
# --------------------------------------------------------------------------- #
def test_beating_cash_is_not_enough_it_must_beat_a_matched_baseline():
    """A strategy whose Sharpe merely ties a constant position fails."""
    r = _everything_passing(
        oos_stats={"sharpe": 0.92, "max_drawdown": -0.51, "n_trades": 572, "sortino": 1.11},
        matched_baseline_stats={"sharpe": 0.90, "sortino": 1.25, "final_equity": 660.0},
    )
    names = [c.name for c in r.blockers]
    assert "beats_matched_baseline" in names
    # Sharpe is ahead but sortino is behind - that is not a win
    detail = next(c.detail for c in r.checks if c.name == "beats_matched_baseline")
    assert "sortino" in detail


def test_one_good_regime_is_not_enough():
    r = _everything_passing(
        regime_report=pd.DataFrame({"dimension": ["trend"] * 3,
                                    "regime": ["bull", "bear", "sideways"],
                                    "excess": [-0.5, 0.4, -0.2]})
    )
    assert "regime_robustness" in [c.name for c in r.blockers]


def test_cross_asset_needs_half_the_unseen_assets():
    r = _everything_passing(
        cross_asset=pd.DataFrame({"tier": ["unseen"] * 4,
                                  "beats_buy_hold": [False, False, False, True]})
    )
    assert "cross_asset" in [c.name for c in r.blockers]
    detail = next(c.detail for c in r.checks if c.name == "cross_asset")
    assert "1/4" in detail


def test_a_deflated_sharpe_below_a_coin_flip_fails():
    r = _everything_passing(deflated_sharpe=0.13)
    assert "statistical_evidence" in [c.name for c in r.blockers]


def test_a_weak_null_p_value_fails():
    r = _everything_passing(
        null_comparison=pd.DataFrame({"null": ["a", "b"], "p_value": [0.02, 0.67]})
    )
    assert "statistical_evidence" in [c.name for c in r.blockers]


def test_cost_robustness_needs_to_survive_the_high_scenario():
    r = _everything_passing(
        cost_ladder=pd.DataFrame({"bps_per_side": [0, 15, 35, 75],
                                  "final_equity": [3000.0, 900.0, 700.0, 400.0]}),
        matched_baseline_stats={"sharpe": 0.6, "sortino": 0.8, "final_equity": 800.0},
    )
    assert "cost_robustness" in [c.name for c in r.blockers]


def test_unrealistic_participation_fails():
    r = _everything_passing(capacity={"max_participation": 0.35})
    assert "capacity" in [c.name for c in r.blockers]


def test_failing_parity_fails_the_gate():
    r = _everything_passing(parity_tests_pass=False)
    assert "execution_realism" in [c.name for c in r.blockers]


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_the_report_says_rejection_is_a_successful_outcome():
    text = evaluate().report()
    assert "NOT ACCEPTED" in text
    assert "successful outcome" in text


def test_the_frame_has_one_row_per_check():
    r = _everything_passing()
    frame = r.frame()
    assert len(frame) == len(r.checks)
    assert set(frame["status"]) <= {"PASS", "FAIL", "NOT RUN"}
