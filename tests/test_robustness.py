"""Tests for the robustness tooling: stability, regimes and the battery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagent.robustness import (
    DEFAULT_ACCEPTANCE,
    as_utc,
    check_acceptance,
    classify_regimes,
    era_report,
    neighbourhood,
    parameter_stability,
    regime_report,
    robustness_battery,
)


# --------------------------------------------------------------------------- #
# neighbourhoods
# --------------------------------------------------------------------------- #
def test_integer_neighbourhood_stays_integral_and_positive():
    out = neighbourhood(50)
    assert all(isinstance(v, int) for v in out)
    assert 50 in out and min(out) > 0 and len(out) >= 3


def test_small_integer_still_gets_neighbours():
    assert neighbourhood(2) == sorted(set(neighbourhood(2)))
    assert len(neighbourhood(1)) >= 2


def test_float_neighbourhood_is_proportional():
    out = neighbourhood(0.2, relative=0.5, n_points=3)
    assert min(out) == pytest.approx(0.1)
    assert max(out) == pytest.approx(0.3)


def test_zero_float_has_no_proportional_neighbours():
    assert neighbourhood(0.0) == [0.0]


def test_boolean_and_categorical_neighbourhoods():
    assert set(neighbourhood(True)) == {False, True}
    assert neighbourhood("adaptive") == ["adaptive"]


# --------------------------------------------------------------------------- #
# parameter stability
# --------------------------------------------------------------------------- #
def test_a_plateau_scores_near_one():
    """Performance that ignores the parameter should score as a plateau."""
    def evaluate(params):
        return {"sharpe": 1.0}          # flat everywhere

    out = parameter_stability(evaluate, {"lookback": 50}, verbose=False)
    assert out["stability"].iloc[0] == pytest.approx(1.0)
    assert out["verdict"].iloc[0] == "plateau"


def test_a_spike_is_flagged():
    """The failure this exists to catch: great at 50, useless at 49."""
    def evaluate(params):
        return {"sharpe": 2.0 if params["lookback"] == 50 else 0.1}

    out = parameter_stability(evaluate, {"lookback": 50}, verbose=False)
    assert out["stability"].iloc[0] < 0.2
    assert "SPIKE" in out["verdict"].iloc[0]


def test_a_neighbour_beating_the_selection_is_called_out():
    def evaluate(params):
        return {"sharpe": 0.3 if params["lookback"] == 50 else 1.5}

    out = parameter_stability(evaluate, {"lookback": 50}, verbose=False)
    assert "neighbour beats it" in out["verdict"].iloc[0]


def test_every_parameter_is_perturbed_independently():
    seen = []

    def evaluate(params):
        seen.append(dict(params))
        return {"sharpe": 1.0}

    parameter_stability(evaluate, {"a": 10, "b": 0.5, "c": 20}, verbose=False)
    # each trial changes exactly one parameter from the selection
    base = {"a": 10, "b": 0.5, "c": 20}
    for trial in seen[1:]:
        differences = [k for k in base if trial[k] != base[k]]
        assert len(differences) <= 1


def test_stability_survives_an_evaluation_that_raises():
    def evaluate(params):
        if params["lookback"] == 62:
            raise RuntimeError("bad cell")
        return {"sharpe": 1.0}

    out = parameter_stability(evaluate, {"lookback": 50}, verbose=False)
    assert not out.empty      # the bad cell is dropped, the row survives


def test_explicit_grids_override_the_inferred_neighbourhood():
    tried = []

    def evaluate(params):
        tried.append(params["factor"])
        return {"sharpe": 1.0}

    parameter_stability(evaluate, {"factor": "a"}, grids={"factor": ["a", "b", "c"]},
                        verbose=False)
    assert set(tried) == {"a", "b", "c"}


# --------------------------------------------------------------------------- #
# regimes
# --------------------------------------------------------------------------- #
@pytest.fixture
def regime_prices():
    idx = pd.date_range("2015-01-01", periods=1400, freq="1D", tz="UTC")
    path = np.concatenate([
        np.linspace(100, 220, 500),      # bull
        np.linspace(220, 120, 300),      # bear
        np.linspace(120, 130, 600),      # sideways
    ])
    return pd.Series(path, index=idx)


def test_regime_labels_identify_bull_bear_and_sideways(regime_prices):
    regimes = classify_regimes(regime_prices, trend_lookback=200)
    labels = set(regimes["trend"].unique())
    assert {"bull", "bear", "sideways"} <= labels


def test_regime_labels_are_causal(regime_prices):
    """A label must not change when later data changes."""
    cut = 900
    full = classify_regimes(regime_prices)["trend"].iloc[:cut]
    truncated = classify_regimes(regime_prices.iloc[:cut])["trend"]
    pd.testing.assert_series_equal(full, truncated, check_names=False)


def test_warmup_bars_are_labelled_not_guessed(regime_prices):
    regimes = classify_regimes(regime_prices, trend_lookback=200)
    assert (regimes["trend"].iloc[:199] == "warmup").all()


def test_regime_report_splits_performance(regime_prices):
    equity = pd.Series(
        100 * np.cumprod(1 + np.random.default_rng(0).normal(0.0003, 0.01, len(regime_prices))),
        index=regime_prices.index,
    )
    report = regime_report(equity, classify_regimes(regime_prices), benchmark=regime_prices)
    assert {"dimension", "regime", "bars", "total_return", "sharpe"} <= set(report.columns)
    assert "excess" in report.columns
    assert (report["share_of_sample"] <= 1.0).all()
    assert "warmup" not in set(report["regime"])


def test_era_report_splits_by_year(regime_prices):
    equity = pd.Series(np.linspace(100, 200, len(regime_prices)), index=regime_prices.index)
    eras = era_report(equity)
    assert len(eras) >= 3
    # A subset check, not an exact one: era_report gained drawdown depth and
    # duration, sortino and win rate so that a regime claim can actually be
    # falsified from it, and pinning the exact column set would make every such
    # addition a test failure rather than the improvement it is.
    assert {"era", "bars", "return", "sharpe", "worst_bar"} <= set(eras.columns)


# --------------------------------------------------------------------------- #
# acceptance and the battery
# --------------------------------------------------------------------------- #
def test_acceptance_criteria_reject_a_weak_result():
    passed, failures = check_acceptance(
        {"sharpe": 0.1, "max_drawdown": -0.6, "n_trades": 5}
    )
    assert not passed
    assert len(failures) == 3           # all three floors breached


def test_acceptance_criteria_accept_a_reasonable_result():
    passed, failures = check_acceptance(
        {"sharpe": 0.9, "max_drawdown": -0.25, "n_trades": 300}
    )
    assert passed and not failures


def test_battery_runs_every_scenario_and_reports_pass_rate():
    def run(**kwargs):
        return {"sharpe": 0.9, "max_drawdown": -0.2, "n_trades": 100,
                "final_equity": 250.0, "total_costs": 10.0}

    out = robustness_battery(run, seeds=(1, 2), start_dates=(None, "2022-01-01"),
                             verbose=False)
    assert len(out) >= 8
    assert out["passed"].all()
    assert "base" in set(out["scenario"])
    assert any(s.startswith("costs:") for s in out["scenario"])
    assert any(s.startswith("spread:") for s in out["scenario"])


def test_battery_records_a_failing_scenario_rather_than_stopping():
    def run(**kwargs):
        if kwargs.get("costs") is not None and kwargs["costs"].name == "high":
            return {"sharpe": -0.5, "max_drawdown": -0.7, "n_trades": 4,
                    "final_equity": 40.0}
        return {"sharpe": 1.0, "max_drawdown": -0.2, "n_trades": 100, "final_equity": 300.0}

    out = robustness_battery(run, seeds=(), verbose=False)
    failed = out[~out["passed"]]
    assert len(failed) == 1
    assert failed.iloc[0]["scenario"] == "costs:high"
    assert failed.iloc[0]["why_failed"]


def test_battery_turns_an_exception_into_a_failed_row():
    def run(**kwargs):
        if kwargs.get("fill_at") == "next_close":
            raise ValueError("boom")
        return {"sharpe": 1.0, "max_drawdown": -0.1, "n_trades": 50, "final_equity": 200.0}

    out = robustness_battery(run, seeds=(), verbose=False)
    row = out[out["scenario"] == "fill:next_close"].iloc[0]
    assert not row["passed"] and "errored" in row["why_failed"]


def test_as_utc_normalises_naive_dates():
    """A naive date slicing a tz-aware index raises; this is why it exists."""
    assert as_utc(None) is None
    assert str(as_utc("2024-01-01").tz) == "UTC"
    assert str(as_utc(pd.Timestamp("2024-01-01", tz="UTC")).tz) == "UTC"


# --------------------------------------------------------------------------- #
# era reporting: enough columns to falsify a regime claim
# --------------------------------------------------------------------------- #
def test_longest_drawdown_bars_counts_the_underwater_run():
    from tradingagent.robustness import longest_drawdown_bars

    idx = pd.date_range("2020-01-01", periods=10, tz="UTC")
    # peak at bar 2, underwater bars 3-7, recovers at 8
    eq = pd.Series([100, 105, 110, 90, 85, 88, 95, 99, 111, 112.0], index=idx)
    assert longest_drawdown_bars(eq) == 5
    assert longest_drawdown_bars(pd.Series(dtype=float)) == 0
    rising = pd.Series(np.arange(1.0, 11.0), index=idx)
    assert longest_drawdown_bars(rising) == 0


def test_era_report_reports_the_benchmark_over_the_same_era():
    """A strategy up 40% while the market rose 45% has not found anything.

    Without a per-era benchmark the report cannot say that, which is why the
    excess column exists.
    """
    from tradingagent.robustness import era_report

    idx = pd.date_range("2019-01-01", periods=800, freq="D", tz="UTC")
    rng = np.random.default_rng(5)
    eq = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 800))), index=idx)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, 800))), index=idx)
    table = era_report(eq, benchmark=bench, periods_per_year=365.0)
    for col in ("benchmark_return", "excess_return", "beat_benchmark",
                "max_drawdown", "drawdown_bars", "sortino", "win_rate_bars"):
        assert col in table.columns, col
    # excess must reconcile with its two inputs
    np.testing.assert_allclose(
        table["excess_return"].to_numpy(),
        (table["return"] - table["benchmark_return"]).to_numpy(),
    )


def test_era_report_works_without_the_optional_inputs():
    """A bare equity curve must still produce a report - the extras are extras."""
    from tradingagent.robustness import era_report

    idx = pd.date_range("2020-01-01", periods=500, freq="D", tz="UTC")
    eq = pd.Series(np.linspace(100, 150, 500), index=idx)
    table = era_report(eq, periods_per_year=365.0)
    assert len(table) >= 1
    assert "benchmark_return" not in table.columns
    assert "max_drawdown" in table.columns
