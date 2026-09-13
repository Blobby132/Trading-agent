"""The CLI is a real entry point, so it gets smoke tests.

These caught a refactor bug where the plotting tail referenced a variable that
was never passed to it - the run printed a full, correct summary and then died
on the last line.
"""

import json
import os

import pytest

from tradingagent.cli import build_parser, main


@pytest.mark.parametrize("mode", ["single", "walkforward"])
def test_cli_runs_end_to_end_offline(tmp_path, mode):
    out = tmp_path / mode
    code = main([
        "--source", "synthetic", "--symbols", "SYN", "--mode", mode,
        "--candidates", "4", "--train-bars", "400", "--test-bars", "200",
        "--capital", "100", "--target", "1000", "--quiet", "--outdir", str(out),
    ])
    assert code == 0
    assert (out / "equity.csv").exists()
    assert (out / "stats.json").exists()
    assert (out / "tearsheet.png").exists(), "the plotting tail never ran"
    stats = json.loads((out / "stats.json").read_text())
    assert stats["initial_equity"] == 100.0
    assert "sharpe" in stats and "max_drawdown" in stats


def test_cli_cross_section_mode_runs_offline(tmp_path):
    out = tmp_path / "xs"
    code = main([
        "--source", "synthetic", "--symbols", *[f"S{i}" for i in range(14)],
        "--mode", "cross-section", "--candidates", "3", "--train-bars", "400",
        "--test-bars", "200", "--capital", "100", "--quiet", "--no-plots",
        "--outdir", str(out),
    ])
    assert code == 0
    assert (out / "equity.csv").exists()


def test_parser_rejects_an_unknown_mode():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--mode", "telepathy"])


def test_cli_robustness_mode_runs_and_saves(tmp_path):
    out = tmp_path / "rob"
    code = main([
        "--source", "synthetic", "--symbols", "SYN", "--mode", "robustness",
        "--capital", "100", "--quiet", "--no-plots", "--outdir", str(out),
    ])
    assert code == 0
    assert (out / "robustness.csv").exists()
    assert (out / "regimes.csv").exists()
    import pandas as pd

    battery = pd.read_csv(out / "robustness.csv")
    assert "base" in set(battery["scenario"])
    assert {"passed", "sharpe", "max_drawdown", "why_failed"} <= set(battery.columns)


def test_cli_cost_scenario_changes_the_result(tmp_path):
    """--cost-scenario must actually reach the engine."""
    import json

    finals = {}
    for scenario in ("low", "high"):
        out = tmp_path / scenario
        assert main([
            "--source", "synthetic", "--symbols", "SYN", "--mode", "single",
            "--cost-scenario", scenario, "--capital", "100", "--quiet",
            "--no-plots", "--outdir", str(out),
        ]) == 0
        finals[scenario] = json.loads((out / "stats.json").read_text())["final_equity"]
    assert finals["low"] > finals["high"], "cost scenario had no effect"
