import json
import os

import pandas as pd
import pytest

from tradingagent.data import synthetic_ohlcv
from tradingagent.holdout import Holdout
from tradingagent.universe import Panel


@pytest.fixture
def ledger(tmp_path):
    return str(tmp_path / "ledger.json")


@pytest.fixture
def frame():
    return synthetic_ohlcv(500, seed=1, start="2020-01-01")


def test_development_and_reserved_do_not_overlap(frame, ledger):
    h = Holdout("2021-01-01", ledger_path=ledger)
    dev, res = h.development(frame), h.reserved(frame)
    assert len(dev) + len(res) == len(frame)
    assert dev.index.max() < h.start <= res.index.min()


def test_development_cannot_see_the_reserved_bars(frame, ledger):
    """The point of hiding it: code built on `development` physically cannot read it."""
    h = Holdout("2021-01-01", ledger_path=ledger)
    assert not h.development(frame).index.isin(h.reserved(frame).index).any()


def test_split_works_on_a_panel(ledger):
    panel = Panel.from_frames({f"S{i}": synthetic_ohlcv(500, seed=i, start="2020-01-01")
                               for i in range(4)})
    h = Holdout("2021-01-01", ledger_path=ledger)
    dev, res = h.development(panel), h.reserved(panel)
    assert isinstance(dev, Panel) and isinstance(res, Panel)
    assert len(dev) + len(res) == len(panel)
    assert set(dev.symbols) == set(panel.symbols)


def test_first_look_is_recorded_and_called_genuine(ledger):
    h = Holdout("2021-01-01", ledger_path=ledger, name="t1")
    assert h.n_uses() == 0
    out = h.evaluate("first try", {"final_equity": 250.0, "sharpe": 1.1,
                                   "max_drawdown": -0.2}, verbose=False)
    assert out["holdout_uses"] == 1.0
    assert "first look" in out["holdout_verdict"]
    assert h.n_uses() == 1


def test_repeated_looks_escalate_the_warning(ledger):
    h = Holdout("2021-01-01", ledger_path=ledger, name="t2")
    verdicts = [
        h.evaluate(f"try {i}", {"final_equity": 100.0 + i, "sharpe": 1.0,
                                "max_drawdown": -0.1}, verbose=False)["holdout_verdict"]
        for i in range(1, 6)
    ]
    assert "first look" in verdicts[0]
    assert "no longer held out" in verdicts[4]
    assert h.n_uses() == 5


def test_the_ledger_survives_a_new_object(ledger):
    Holdout("2021-01-01", ledger_path=ledger, name="t3").evaluate(
        "once", {"final_equity": 100.0, "sharpe": 0.5, "max_drawdown": -0.1}, verbose=False
    )
    assert Holdout("2021-01-01", ledger_path=ledger, name="t3").n_uses() == 1


def test_ledgers_are_kept_separate_by_name(ledger):
    a = Holdout("2021-01-01", ledger_path=ledger, name="alpha")
    b = Holdout("2021-01-01", ledger_path=ledger, name="beta")
    a.evaluate("x", {"final_equity": 1.0, "sharpe": 0.0, "max_drawdown": 0.0}, verbose=False)
    assert a.n_uses() == 1 and b.n_uses() == 0


def test_report_lists_previous_looks(ledger):
    h = Holdout("2021-01-01", ledger_path=ledger, name="t4")
    for i in range(3):
        h.evaluate(f"attempt {i}", {"final_equity": 100.0 * (i + 1), "sharpe": 1.0,
                                    "max_drawdown": -0.1}, verbose=False)
    text = h.report()
    assert "times evaluated   3" in text
    assert "attempt 0" in text


def test_reset_clears_only_its_own_entries(ledger):
    a = Holdout("2021-01-01", ledger_path=ledger, name="keep")
    b = Holdout("2021-01-01", ledger_path=ledger, name="drop")
    a.evaluate("x", {"final_equity": 1.0, "sharpe": 0.0, "max_drawdown": 0.0}, verbose=False)
    b.evaluate("y", {"final_equity": 1.0, "sharpe": 0.0, "max_drawdown": 0.0}, verbose=False)
    b.reset()
    assert a.n_uses() == 1 and b.n_uses() == 0


def test_covers_detects_whether_data_reaches_the_reserved_era(frame, ledger):
    h = Holdout("2021-01-01", ledger_path=ledger)
    assert h.covers(frame)
    assert not h.covers(frame.loc[:"2020-06-01"])
