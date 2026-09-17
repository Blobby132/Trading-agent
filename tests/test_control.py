"""The frozen control must stay frozen.

A control that drifts is not a control. These tests fail if the definition is
edited, which is the point: a later phase that wants a different baseline has to
create a new named control rather than quietly moving this one.
"""

from __future__ import annotations

import pytest

from tradingagent.control import CONTROL_METRICS, CONTROL_NAME, CONTROL_OVERRIDES, CONTROL_SPEC


def test_control_name_is_stable():
    assert CONTROL_NAME == "BTC_FLOOR_050_CONTROL"


def test_control_is_the_trend_floor_and_nothing_else():
    """The control deviates from repository defaults in exactly one place."""
    assert CONTROL_OVERRIDES == {"trend_floor": [0.50]}


def test_control_spec_records_what_a_comparison_needs():
    for key in ("symbol", "interval", "source", "traded_start", "traded_end",
                "cost_scenario", "execution", "search_space_size",
                "candidates_per_fold", "seeds", "initial_capital"):
        assert key in CONTROL_SPEC, key
    assert CONTROL_SPEC["execution"].startswith("decision at close of bar t")
    assert CONTROL_SPEC["search_space_size"] == 2304


def test_control_metrics_are_pinned():
    """If a code change moves the control's numbers, this fails and the change
    has to be justified rather than absorbed."""
    assert CONTROL_METRICS["median_final_equity"] == pytest.approx(2475.29, rel=1e-4)
    assert CONTROL_METRICS["median_sharpe"] == pytest.approx(1.088, rel=1e-3)
    assert CONTROL_METRICS["seed0_final_equity"] == pytest.approx(2479.37, rel=1e-4)


def test_control_safety_statement_mentions_the_interlocks():
    assert "no live trading" in CONTROL_SPEC["safety"]
    assert "paper interlocks untouched" in CONTROL_SPEC["safety"]
