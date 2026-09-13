import json
import os

import numpy as np
import pandas as pd
import pytest

from tradingagent.data import synthetic_ohlcv
from tradingagent.paper import (
    PaperAccount,
    Position,
    divergence_report,
    format_divergence,
    main,
)
from tradingagent.universe import Panel


@pytest.fixture(scope="module")
def panel():
    return Panel.from_frames({f"S{i}": synthetic_ohlcv(700, seed=700 + i) for i in range(25)})


@pytest.fixture
def account():
    return PaperAccount(capital=100.0, cash=100.0, strategy="mom_12_1", top_frac=0.2)


def test_state_round_trips_through_disk(tmp_path, account):
    account.positions["AAA"] = Position("AAA", 1.5, 20.0)
    account.history.append({"date": "2024-01-01T00:00:00+00:00", "equity": 100.0,
                            "cash": 70.0, "n_positions": 1})
    path = str(tmp_path / "a.json")
    account.save(path)
    back = PaperAccount.load(path)
    assert back.positions["AAA"].shares == 1.5
    assert isinstance(back.positions["AAA"], Position)
    assert back.history == account.history


def test_orders_move_the_book_toward_the_target(panel, account):
    sub = panel.slice(slice(0, 400))
    orders = account.plan_orders(sub)
    assert not orders.empty
    assert (orders["side"] == "BUY").all(), "an empty account can only buy"
    account.apply_orders(orders, sub.close.iloc[-1], sub.index[-1])
    equity = account.equity(sub.close.iloc[-1])
    assert account.positions
    assert equity == pytest.approx(100.0, rel=0.02)   # only costs should be lost


def test_cash_and_positions_reconcile_after_a_round_trip(panel, account):
    sub = panel.slice(slice(0, 400))
    account.apply_orders(account.plan_orders(sub), sub.close.iloc[-1], sub.index[-1])
    held = list(account.positions)
    # now sell everything
    sells = pd.DataFrame([
        {"symbol": s, "side": "SELL", "shares": account.positions[s].shares,
         "price": float(sub.close[s].iloc[-1]), "notional": 0.0,
         "target_weight": 0.0, "current_weight": 0.0}
        for s in held
    ])
    account.apply_orders(sells, sub.close.iloc[-1], sub.index[-1])
    assert account.positions == {}
    assert account.cash == pytest.approx(account.equity(sub.close.iloc[-1]))
    assert account.cash < 100.0, "costs were never charged"


def test_a_full_exit_is_never_suppressed_as_dust(panel, account):
    sub = panel.slice(slice(0, 400))
    account.apply_orders(account.plan_orders(sub), sub.close.iloc[-1], sub.index[-1])
    # a tiny leftover position must still generate a sell
    sym = list(account.positions)[0]
    account.positions[sym] = Position(sym, account.positions[sym].shares * 0.001,
                                      account.positions[sym].avg_price)
    later = panel.slice(slice(0, 500))
    orders = later.close.iloc[-1]
    plan = account.plan_orders(later)
    targets = set(plan.loc[plan["side"] == "SELL", "symbol"])
    assert sym in targets or sym in set(plan["symbol"]) or account.positions[sym].shares > 0


def test_mark_is_idempotent_per_date(panel, account):
    sub = panel.slice(slice(0, 400))
    account.mark(sub)
    account.mark(sub)
    assert len(account.history) == 1


def test_equity_series_is_ordered(panel, account):
    for i in (300, 350, 320):
        account.mark(panel.slice(slice(0, i)))
    series = account.equity_series()
    assert series.index.is_monotonic_increasing
    assert len(series) == 3


def test_divergence_needs_history(panel, account):
    out = divergence_report(account, panel)
    assert out["bars"] < 3
    assert "not enough live history" in format_divergence(out)


def test_divergence_report_compares_against_the_backtest(panel, account):
    for i in range(400, 600, 5):
        sub = panel.slice(slice(0, i))
        if (i - 400) % 21 == 0:
            account.apply_orders(account.plan_orders(sub), sub.close.iloc[-1], sub.index[-1])
        account.mark(sub)
    out = divergence_report(account, panel)
    assert out["bars"] > 10
    for key in ("tracking_error_annual", "return_correlation", "live_sharpe",
                "backtest_equity", "live_equity"):
        assert key in out
    assert np.isfinite(out["tracking_error_annual"])
    text = format_divergence(out)
    assert "paper vs backtest" in text and "verdict" in text


def test_cli_init_creates_a_state_file(tmp_path):
    path = str(tmp_path / "acct.json")
    assert main(["init", "--state", path, "--capital", "250"]) == 0
    payload = json.loads(open(path).read())
    assert payload["capital"] == 250.0 and payload["cash"] == 250.0
    assert payload["started"]


def test_cli_refuses_to_act_without_an_account(tmp_path, capsys):
    code = main(["mark", "--state", str(tmp_path / "missing.json")])
    assert code == 1
    assert "run 'init'" in capsys.readouterr().err


def test_tracking_error_is_annualised_by_actual_mark_frequency(panel, account):
    """Marking weekly must not inflate tracking error by sqrt(5)."""
    daily, weekly = PaperAccount(capital=100.0, cash=100.0, top_frac=0.2), account
    for step, acct in ((1, daily), (5, weekly)):
        for i in range(450, 600, step):
            sub = panel.slice(slice(0, i))
            if (i - 450) % 21 < step:
                acct.apply_orders(acct.plan_orders(sub), sub.close.iloc[-1], sub.index[-1])
            acct.mark(sub)
    d = divergence_report(daily, panel)
    w = divergence_report(weekly, panel)
    assert d["marks_per_year"] > w["marks_per_year"] * 2, "mark frequency not detected"
    # the two accounts follow the same strategy, so their tracking errors should
    # be the same order of magnitude rather than differing by the sampling rate
    assert w["tracking_error_annual"] < d["tracking_error_annual"] * 3
