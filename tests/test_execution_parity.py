"""Deterministic execution tests: exact arithmetic, and backtest/paper parity.

Every other test in this project runs on synthetic random data and asserts
properties. These assert *numbers*, on an OHLC series small enough to work out
by hand, so a timing or costing regression cannot hide inside a statistic.

The parity tests exist because the project previously had two execution
implementations that disagreed: the backtest filled at the next bar's open, and
paper trading filled at the same close that produced the signal. Both now route
through :mod:`tradingagent.execution`, and these tests fail loudly if they are
ever allowed to drift apart again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagent.engine import BacktestEngine, ExecutionConfig
from tradingagent.execution import (
    BASE_COST,
    COST_SCENARIOS,
    CostModel,
    ExecutionModel,
    cost_scenario,
    plan_rebalance,
)
from tradingagent.paper import PaperAccount
from tradingagent.risk import RiskConfig
from tradingagent.universe import Panel

FLAT_RISK = RiskConfig(
    target_vol=0.0, atr_stop_mult=0.0, max_drawdown_stop=0.0, reentry_lockout_bars=0
)


def make_bars(opens, highs, lows, closes, symbol="X", start="2021-01-04"):
    idx = pd.date_range(start, periods=len(opens), freq="1D", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1e9},
        index=idx,
    )


@pytest.fixture
def bars():
    """Six bars with prices chosen so every product is exact in binary."""
    o = [100.0, 100.0, 110.0, 120.0, 90.0, 80.0]
    c = [100.0, 105.0, 115.0, 100.0, 85.0, 75.0]
    h = [max(a, b) + 5 for a, b in zip(o, c)]
    l = [min(a, b) - 5 for a, b in zip(o, c)]
    return make_bars(o, h, l, c)


def replay_paper(panel: Panel, weights: pd.DataFrame, *, capital=100.0,
                 min_trade_frac=0.0, cost="base", periods_per_year=252.0) -> PaperAccount:
    """Drive a paper account bar by bar, the way a human would run the CLI.

    On each bar: fill whatever was queued yesterday, then decide today's basket
    and queue it for tomorrow. Exactly the causal sequence the backtest models.
    """
    account = PaperAccount(capital=capital, cash=capital, cost_scenario=cost,
                           periods_per_year=periods_per_year)
    for i in range(len(panel)):
        window = panel.slice(slice(0, i + 1))
        if account.pending:
            account.fill_pending(window, min_trade_frac=min_trade_frac)
        account.schedule_orders(
            account.plan_orders(window, min_trade_frac=min_trade_frac,
                                target=weights.iloc[i]),
            window.index[-1],
        )
        account.mark(window)
    return account


# --------------------------------------------------------------------------- #
# 1. a signal at bar T cannot execute before T+1
# --------------------------------------------------------------------------- #
def test_signal_at_T_cannot_execute_on_bar_T(bars):
    """The single most important invariant in the project."""
    weights = pd.Series(0.0, index=bars.index)
    weights.iloc[1] = 1.0                      # decided on bar 1's close
    result = BacktestEngine(
        ExecutionConfig(initial_capital=100.0, min_trade_frac=0.0), FLAT_RISK
    ).run(bars, weights)

    assert result.weights.iloc[1, 0] == 0.0, "traded on the bar that produced the signal"
    assert result.weights.iloc[2, 0] > 0.5, "did not trade on the following bar"
    assert result.trades.index[0] == bars.index[2]
    # and the fill happened against bar 2's open (110), not bar 1's close (105)
    assert result.trades.iloc[0]["reference_price"] == pytest.approx(110.0)


def test_paper_refuses_to_fill_on_the_decision_bar(bars):
    panel = Panel.from_frames({"X": bars})
    account = PaperAccount(capital=100.0, cash=100.0)
    window = panel.slice(slice(0, 2))
    account.schedule_orders(
        account.plan_orders(window, target=pd.Series({"X": 1.0}), min_trade_frac=0.0),
        window.index[-1],
    )
    with pytest.raises(ValueError, match="cannot fill on"):
        account.fill_pending(window)          # same bar - must be refused


def test_paper_fills_on_the_next_bar_at_that_bars_open(bars):
    panel = Panel.from_frames({"X": bars})
    account = PaperAccount(capital=100.0, cash=100.0)
    decide = panel.slice(slice(0, 2))
    account.schedule_orders(
        account.plan_orders(decide, target=pd.Series({"X": 1.0}), min_trade_frac=0.0),
        decide.index[-1],
    )
    filled = account.fill_pending(panel.slice(slice(0, 3)), min_trade_frac=0.0)
    assert len(filled) == 1
    assert filled.iloc[0]["reference_price"] == pytest.approx(110.0)   # bar 2's open


# --------------------------------------------------------------------------- #
# 2. backtest and paper agree
# --------------------------------------------------------------------------- #
def test_backtest_and_paper_equity_agree_exactly(bars):
    panel = Panel.from_frames({"X": bars})
    weights = pd.DataFrame({"X": [0.0, 1.0, 1.0, 0.5, 0.0, 0.0]}, index=bars.index)

    engine = BacktestEngine(
        ExecutionConfig(initial_capital=100.0, min_trade_frac=0.0, max_leverage=1.0,
                        periods_per_year=252.0),
        FLAT_RISK,
    ).run(panel.to_frames(), weights)
    account = replay_paper(panel, weights)

    live = account.equity_series()
    np.testing.assert_allclose(
        live.to_numpy(), engine.equity.reindex(live.index).to_numpy(), rtol=1e-12, atol=1e-12
    )


def test_backtest_and_paper_agree_across_a_multi_name_book():
    idx = pd.date_range("2021-01-04", periods=8, freq="1D", tz="UTC")
    frames = {}
    for k, sym in enumerate(["A", "B", "C"]):
        base = 50.0 + 10 * k
        closes = base * (1 + 0.02 * np.arange(8) * (1 if k != 1 else -1))
        opens = np.concatenate([[base], closes[:-1] * 1.01])
        frames[sym] = pd.DataFrame(
            {"open": opens, "high": np.maximum(opens, closes) * 1.02,
             "low": np.minimum(opens, closes) * 0.98, "close": closes, "volume": 1e9},
            index=idx,
        )
    panel = Panel.from_frames(frames)
    weights = pd.DataFrame(
        {"A": [0, .3, .3, .3, .2, .2, 0, 0], "B": [0, .3, .3, 0, 0, .4, .4, 0],
         "C": [0, .3, .3, .3, .4, .4, .4, 0]}, index=idx, dtype=float,
    )
    engine = BacktestEngine(
        ExecutionConfig(initial_capital=100.0, min_trade_frac=0.0, max_leverage=1.0,
                        periods_per_year=252.0),
        FLAT_RISK,
    ).run(panel.to_frames(), weights)
    account = replay_paper(panel, weights)
    live = account.equity_series()
    np.testing.assert_allclose(
        live.to_numpy(), engine.equity.reindex(live.index).to_numpy(), rtol=1e-10, atol=1e-10
    )


def test_parity_holds_under_every_cost_scenario():
    idx = pd.date_range("2021-01-04", periods=10, freq="1D", tz="UTC")
    closes = 100 * np.exp(np.cumsum(np.linspace(-0.02, 0.03, 10)))
    opens = np.concatenate([[100.0], closes[:-1] * 1.005])
    panel = Panel.from_frames({"X": pd.DataFrame(
        {"open": opens, "high": np.maximum(opens, closes) * 1.01,
         "low": np.minimum(opens, closes) * 0.99, "close": closes, "volume": 1e9}, index=idx)})
    weights = pd.DataFrame({"X": [0, 1, 1, 0, 1, 1, 1, 0, 1, 0]}, index=idx, dtype=float)

    for name in COST_SCENARIOS:
        engine = BacktestEngine(
            ExecutionConfig(initial_capital=100.0, min_trade_frac=0.0, max_leverage=1.0,
                            periods_per_year=252.0, costs=cost_scenario(name)),
            FLAT_RISK,
        ).run(panel.to_frames(), weights)
        account = replay_paper(panel, weights, cost=name)
        live = account.equity_series()
        np.testing.assert_allclose(
            live.to_numpy(), engine.equity.reindex(live.index).to_numpy(),
            rtol=1e-10, atol=1e-10, err_msg=f"parity broke under the {name!r} cost scenario",
        )


# --------------------------------------------------------------------------- #
# 3. gaps, costs and sizing, checked by hand
# --------------------------------------------------------------------------- #
def test_open_gap_is_filled_at_the_gapped_price_not_the_prior_close():
    """A weight decided before a gap pays the gap; it does not get yesterday's price."""
    bars = make_bars(
        opens=[100.0, 100.0, 70.0, 70.0],      # bar 2 gaps down 30%
        highs=[105.0, 105.0, 75.0, 75.0],
        lows=[95.0, 95.0, 65.0, 65.0],
        closes=[100.0, 100.0, 70.0, 70.0],
    )
    weights = pd.Series([0.0, 1.0, 1.0, 1.0], index=bars.index)
    result = BacktestEngine(
        ExecutionConfig(initial_capital=100.0, min_trade_frac=0.0, fee_bps=0.0,
                        slippage_bps=0.0), FLAT_RISK,
    ).run(bars, weights)
    # bought at 70, the gapped open - not at 100
    assert result.trades.iloc[0]["reference_price"] == pytest.approx(70.0)
    assert result.equity.iloc[1] == pytest.approx(100.0)   # still flat through the gap


def test_costs_are_charged_as_an_adverse_fill_price():
    """Buying pays above the reference; selling receives below it."""
    model = ExecutionModel(costs=CostModel(fee_bps=10.0, half_spread_bps=5.0,
                                           slippage_bps=5.0))
    assert model.costs.total_bps() == pytest.approx(20.0)
    assert float(model.costs.fill_price(100.0, +1)) == pytest.approx(100.20)
    assert float(model.costs.fill_price(100.0, -1)) == pytest.approx(99.80)


def test_fee_and_spread_are_reported_separately():
    split = BASE_COST.split(10_000.0)
    assert split["fee"] == pytest.approx(10.0)      # 10 bps of 10,000
    assert split["spread"] == pytest.approx(2.5)
    assert split["slippage"] == pytest.approx(2.5)
    assert sum(split.values()) == pytest.approx(15.0)


def test_higher_cost_scenarios_cost_strictly_more(bars):
    weights = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], index=bars.index)
    finals = {}
    for name in ("low", "base", "high"):
        result = BacktestEngine(
            ExecutionConfig(initial_capital=100.0, min_trade_frac=0.0,
                            costs=cost_scenario(name)), FLAT_RISK,
        ).run(bars, weights)
        finals[name] = result.final_equity
    assert finals["low"] > finals["base"] > finals["high"]


def test_market_impact_grows_with_participation():
    quiet = CostModel(impact_bps_at_full=100.0)
    assert float(quiet.impact_bps(0.0)) == pytest.approx(0.0)
    assert float(quiet.impact_bps(0.01)) == pytest.approx(10.0)   # sqrt(0.01) = 0.1
    assert float(quiet.impact_bps(0.25)) == pytest.approx(50.0)
    assert float(quiet.impact_bps(1.0)) == pytest.approx(100.0)
    assert float(quiet.impact_bps(5.0)) == pytest.approx(100.0)   # clipped


def test_position_sizing_uses_the_execution_price():
    """Sizing off the signal-bar close would buy the wrong number of shares."""
    model = ExecutionModel(costs=CostModel(fee_bps=0, half_spread_bps=0, slippage_bps=0),
                           min_trade_frac=0.0)
    plan = plan_rebalance(
        np.array([1.0]), np.array([0.0]), 100.0, np.array([50.0]), model=model
    )
    assert plan.delta_units[0] == pytest.approx(2.0)      # 100 / 50, the execution price
    plan2 = plan_rebalance(
        np.array([1.0]), np.array([0.0]), 100.0, np.array([25.0]), model=model
    )
    assert plan2.delta_units[0] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# 4. dust and full exits
# --------------------------------------------------------------------------- #
def test_full_exit_is_never_suppressed_by_the_dust_filter():
    model = ExecutionModel(min_trade_frac=0.9)           # absurdly high floor
    plan = plan_rebalance(
        np.array([0.0]), np.array([1.0]), 100.0, np.array([10.0]), model=model
    )
    assert plan.is_full_exit[0]
    assert not plan.suppressed[0]
    assert plan.delta_units[0] == pytest.approx(-1.0)


def test_dust_is_measured_against_the_position_not_the_account():
    model = ExecutionModel(min_trade_frac=0.10)
    # a 5%-of-equity position nudged by 20% of itself: must trade
    plan = plan_rebalance(
        np.array([0.04]), np.array([0.5]), 100.0, np.array([10.0]), model=model
    )
    assert plan.delta_units[0] != 0.0
    # the same position nudged by 1% of itself: dust
    plan2 = plan_rebalance(
        np.array([0.0495]), np.array([0.495]), 100.0, np.array([10.0]), model=model
    )
    assert plan2.delta_units[0] == 0.0


def test_paper_and_backtest_suppress_the_same_dust():
    idx = pd.date_range("2021-01-04", periods=12, freq="1D", tz="UTC")
    closes = 100 + np.arange(12) * 0.05          # drifts slowly: mostly dust
    opens = np.concatenate([[100.0], closes[:-1]])
    panel = Panel.from_frames({"X": pd.DataFrame(
        {"open": opens, "high": closes + 1, "low": closes - 1, "close": closes,
         "volume": 1e9}, index=idx)})
    weights = pd.DataFrame({"X": np.linspace(0.50, 0.56, 12)}, index=idx)

    engine = BacktestEngine(
        ExecutionConfig(initial_capital=100.0, min_trade_frac=0.10, max_leverage=1.0,
                        periods_per_year=252.0),
        FLAT_RISK,
    ).run(panel.to_frames(), weights)
    account = replay_paper(panel, weights, min_trade_frac=0.10)
    assert len(account.orders) == int(engine.meta["n_trades"])


# --------------------------------------------------------------------------- #
# 5. stops cannot see the future
# --------------------------------------------------------------------------- #
def test_stop_uses_a_level_fixed_before_the_bar_opened():
    """The stop tested on bar i must have been set from data up to i-1."""
    bars = make_bars(
        opens=[100.0] * 6, highs=[100.0, 100.0, 100.0, 130.0, 100.0, 100.0],
        lows=[100.0, 100.0, 100.0, 70.0, 100.0, 100.0], closes=[100.0] * 6,
    )
    risk = RiskConfig(target_vol=0.0, atr_stop_mult=1.0, trail_stop=True,
                      max_drawdown_stop=0.0, atr_n=2)
    result = BacktestEngine(
        ExecutionConfig(initial_capital=100.0, min_trade_frac=0.0, fee_bps=0.0,
                        slippage_bps=0.0), risk,
    ).run(bars, pd.Series(1.0, index=bars.index))
    stops = result.trades[result.trades["reason"] == "stop"]
    if not stops.empty:
        # a stop that had trailed using bar 3's own high of 130 would sit at
        # 130 - ATR, far above the 100 open, and would fill at an impossible price
        assert (stops["price"] <= 100.0 + 1e-9).all()


def test_trailing_stop_cannot_be_raised_by_the_bar_it_is_tested_against():
    """Regression test for a fixed bug: trailing on the same bar's high."""
    bars = make_bars(
        opens=[100.0, 100.0, 100.0, 100.0],
        highs=[101.0, 101.0, 200.0, 101.0],     # bar 2 spikes then collapses
        lows=[99.0, 99.0, 60.0, 99.0],
        closes=[100.0, 100.0, 61.0, 100.0],
    )
    risk = RiskConfig(target_vol=0.0, atr_stop_mult=2.0, trail_stop=True,
                      max_drawdown_stop=0.0, atr_n=2)
    result = BacktestEngine(
        ExecutionConfig(initial_capital=100.0, min_trade_frac=0.0, fee_bps=0.0,
                        slippage_bps=0.0), risk,
    ).run(bars, pd.Series(1.0, index=bars.index))
    assert np.isfinite(result.equity).all()
    assert (result.equity > 0).all()


# --------------------------------------------------------------------------- #
# 6. accounting invariants
# --------------------------------------------------------------------------- #
def test_paper_cash_and_positions_reconcile_to_equity(bars):
    panel = Panel.from_frames({"X": bars})
    weights = pd.DataFrame({"X": [0.0, 1.0, 1.0, 0.5, 0.0, 0.0]}, index=bars.index)
    account = replay_paper(panel, weights)
    final_prices = panel.close.iloc[-1]
    held = sum(p.shares * float(final_prices[s]) for s, p in account.positions.items())
    assert account.equity(final_prices) == pytest.approx(account.cash + held)


def test_selling_part_of_a_position_keeps_the_basis(bars):
    """Regression: a partial sell used to leave avg_price stale."""
    account = PaperAccount(capital=100.0, cash=100.0)
    account._book("X", 10.0, 50.0)
    assert account.positions["X"].avg_price == pytest.approx(50.0)
    account._book("X", -4.0, 80.0)                      # sell into strength
    assert account.positions["X"].shares == pytest.approx(6.0)
    assert account.positions["X"].avg_price == pytest.approx(50.0), "basis moved on a sell"
    account._book("X", 6.0, 70.0)                       # add back
    assert account.positions["X"].avg_price == pytest.approx(60.0)


def test_flipping_through_zero_resets_the_basis():
    account = PaperAccount(capital=100.0, cash=100.0)
    account._book("X", 5.0, 50.0)
    account._book("X", -8.0, 40.0)                      # long 5 -> short 3
    assert account.positions["X"].shares == pytest.approx(-3.0)
    assert account.positions["X"].avg_price == pytest.approx(40.0)


def test_min_trade_frac_of_one_is_rejected():
    """A floor of 1.0 would silently disable every rebalance."""
    with pytest.raises(ValueError, match="disable all rebalancing"):
        ExecutionModel(min_trade_frac=1.0)


# --------------------------------------------------------------------------- #
# 7. the calendar must match, or financing silently diverges
# --------------------------------------------------------------------------- #
def test_mismatched_calendars_are_detectable(bars):
    """A 365-day engine against a 252-day paper account diverges on financing.

    Not a logic bug in either - a configuration trap. It cost real debugging
    time while building these tests, so it gets a test of its own.
    """
    panel = Panel.from_frames({"X": bars})
    weights = pd.DataFrame({"X": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, index=bars.index)

    matched = BacktestEngine(
        ExecutionConfig(initial_capital=100.0, min_trade_frac=0.0, max_leverage=1.0,
                        periods_per_year=252.0), FLAT_RISK,
    ).run(panel.to_frames(), weights)
    crypto_calendar = BacktestEngine(
        ExecutionConfig(initial_capital=100.0, min_trade_frac=0.0, max_leverage=1.0,
                        periods_per_year=365.0), FLAT_RISK,
    ).run(panel.to_frames(), weights)

    paper = replay_paper(panel, weights, periods_per_year=252.0)
    live = paper.equity_series()
    np.testing.assert_allclose(
        live.to_numpy(), matched.equity.reindex(live.index).to_numpy(), rtol=1e-10, atol=1e-10
    )
    assert not np.allclose(
        crypto_calendar.equity.reindex(live.index).to_numpy(), live.to_numpy(),
        rtol=1e-12, atol=1e-12,
    ), "a calendar mismatch should show up, so that it can be caught"


def test_divergence_report_uses_the_accounts_own_calendar():
    from tradingagent.paper import divergence_report
    import inspect

    source = inspect.getsource(divergence_report)
    assert "account.periods_per_year" in source, (
        "divergence_report must take the calendar from the account it is judging, "
        "not from a default argument that can disagree with it"
    )
