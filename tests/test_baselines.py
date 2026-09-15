"""Baselines: simple, unsearched, and run through the same engine as the agent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagent.baselines import (
    BASELINES,
    baseline_table,
    buy_and_hold,
    cash,
    excess_over_baselines,
    price_above_sma,
    run_baseline,
    simple_momentum,
    sma_crossover,
)
from tradingagent.data import synthetic_ohlcv
from tradingagent.engine import ExecutionConfig


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return synthetic_ohlcv(1500, seed=21)


def test_every_baseline_produces_a_weight_in_range(data):
    for name, fn in BASELINES.items():
        w = fn(data)
        assert len(w) == len(data), name
        assert w.between(-1.0, 1.0).all(), name
        assert not w.isna().any(), name


def test_cash_holds_nothing_and_costs_nothing(data):
    stats = run_baseline("cash", data, base_exec=ExecutionConfig(initial_capital=100.0))
    assert stats["final_equity"] == pytest.approx(100.0)
    assert stats["n_trades"] == 0
    assert stats["total_costs"] == pytest.approx(0.0)


def test_buy_and_hold_trades_once_and_stays_invested(data):
    stats = run_baseline("buy_and_hold", data, base_exec=ExecutionConfig(initial_capital=100.0))
    assert stats["n_trades"] == 1
    assert stats["time_in_market"] > 0.98


def test_baselines_are_long_only(data):
    """A $100 cash account cannot short, so a baseline that shorts is not a
    comparable alternative."""
    for name, fn in BASELINES.items():
        assert (fn(data) >= 0.0).all(), name


def test_baselines_have_no_lookahead(data):
    """Same poisoning treatment as everything else in the package."""
    cut = 900
    dirty = data.copy()
    rng = np.random.default_rng(0)
    shock = 500.0 * np.exp(np.cumsum(rng.normal(-0.01, 0.15, size=len(data) - cut)))
    for col, mult in (("close", 1.0), ("open", 1.01), ("high", 1.05), ("low", 0.95)):
        dirty.iloc[cut:, dirty.columns.get_loc(col)] = shock * mult
    for name, fn in BASELINES.items():
        pd.testing.assert_series_equal(
            fn(data).iloc[:cut], fn(dirty).iloc[:cut], rtol=0, atol=0, obj=name
        )


def test_sma_crossover_is_flat_before_its_window_fills(data):
    w = sma_crossover(data, 50, 200)
    assert (w.iloc[:199] == 0.0).all()


def test_momentum_is_flat_before_its_window_fills(data):
    assert (simple_momentum(data, 252).iloc[:251] == 0.0).all()


def test_baseline_table_covers_every_baseline(data):
    table = baseline_table(data, base_exec=ExecutionConfig(initial_capital=100.0))
    assert set(table["baseline"]) == set(BASELINES)
    assert table["final_equity"].notna().all()


def test_baselines_pay_real_costs(data):
    """A baseline with turnover must be charged for it, or the comparison is rigged
    in the baseline's favour."""
    cheap = run_baseline("sma_50_200", data,
                         base_exec=ExecutionConfig(initial_capital=100.0, fee_bps=0.0, slippage_bps=0.0))
    dear = run_baseline("sma_50_200", data,
                        base_exec=ExecutionConfig(initial_capital=100.0, fee_bps=100.0, slippage_bps=100.0))
    if cheap["n_trades"] > 0:
        assert dear["final_equity"] < cheap["final_equity"]
        assert dear["total_costs"] > cheap["total_costs"]


def test_excess_over_baselines_flags_a_loss(data):
    table = baseline_table(data, base_exec=ExecutionConfig(initial_capital=100.0))
    out = excess_over_baselines({"final_equity": 0.0}, table)
    assert not out["beats"].any()
    out2 = excess_over_baselines({"final_equity": 1e9}, table)
    assert out2["beats"].all()
