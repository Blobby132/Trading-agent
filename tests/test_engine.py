import numpy as np
import pandas as pd
import pytest

from tradingagent.engine import BacktestEngine, ExecutionConfig, buy_and_hold_equity
from tradingagent.risk import RiskConfig

FRICTIONLESS = ExecutionConfig(
    initial_capital=100.0, fee_bps=0.0, slippage_bps=0.0, borrow_rate=0.0,
    short_rate=0.0, min_trade_frac=0.0,
)
NO_RISK_LAYER = RiskConfig(
    target_vol=0.0, atr_stop_mult=0.0, max_drawdown_stop=0.0, reentry_lockout_bars=0
)


def test_full_long_tracks_the_asset(trending_prices):
    """Frictionless, fully invested: equity must track the price 1-for-1."""
    w = pd.Series(1.0, index=trending_prices.index)
    res = BacktestEngine(FRICTIONLESS, NO_RISK_LAYER).run(trending_prices, w)
    # entry is at the open of bar 1 (the signal at bar 0 is filled next bar)
    entry = trending_prices["open"].iloc[1]
    expected = 100.0 * trending_prices["close"].iloc[-1] / entry
    assert res.final_equity == pytest.approx(expected, rel=1e-9)


def test_signal_is_filled_on_the_following_bar(trending_prices):
    """A signal that appears at bar t must not be tradeable during bar t."""
    w = pd.Series(0.0, index=trending_prices.index)
    w.iloc[10] = 1.0
    res = BacktestEngine(FRICTIONLESS, NO_RISK_LAYER).run(trending_prices, w)
    assert res.weights.iloc[10, 0] == pytest.approx(0.0)   # still flat on the signal bar
    assert res.weights.iloc[11, 0] > 0.5                   # in the market the bar after
    assert res.trades.index[0] == trending_prices.index[11]


def test_costs_only_ever_reduce_equity(trending_prices):
    w = pd.Series(1.0, index=trending_prices.index)
    free = BacktestEngine(FRICTIONLESS, NO_RISK_LAYER).run(trending_prices, w)
    costly = BacktestEngine(
        ExecutionConfig(initial_capital=100.0, fee_bps=20.0, slippage_bps=10.0, min_trade_frac=0.0),
        NO_RISK_LAYER,
    ).run(trending_prices, w)
    assert costly.final_equity < free.final_equity
    assert costly.costs["fees"].sum() > 0


def test_short_position_profits_when_price_falls():
    idx = pd.date_range("2021-01-01", periods=50, freq="1D", tz="UTC")
    close = pd.Series(100.0 * (0.99 ** np.arange(50)), index=idx)
    df = pd.DataFrame(
        {"open": close.shift(1).fillna(100.0), "high": close * 1.001,
         "low": close * 0.999, "close": close, "volume": 1.0}
    )
    res = BacktestEngine(FRICTIONLESS, NO_RISK_LAYER).run(df, pd.Series(-1.0, index=idx))
    assert res.final_equity > 100.0


def test_gross_exposure_never_exceeds_the_cap(prices):
    cfg = ExecutionConfig(initial_capital=100.0, max_leverage=2.0, min_trade_frac=0.0)
    res = BacktestEngine(cfg, NO_RISK_LAYER).run(prices, pd.Series(5.0, index=prices.index))
    # the cap binds at the moment of the fill; intrabar drift can push the
    # marked exposure slightly past it before the next rebalance
    assert res.weights.abs().sum(axis=1).max() <= 2.0 * 1.25


def test_stop_loss_closes_the_position():
    idx = pd.date_range("2021-01-01", periods=40, freq="1D", tz="UTC")
    close = pd.Series(100.0, index=idx)
    close.iloc[25:] = 50.0       # a crash the stop has to catch
    df = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1.0}
    )
    risk = RiskConfig(target_vol=0.0, atr_stop_mult=1.0, trail_stop=False, max_drawdown_stop=0.0)
    res = BacktestEngine(FRICTIONLESS, risk).run(df, pd.Series(1.0, index=idx))
    assert (res.trades["reason"] == "stop").any()


def test_drawdown_kill_switch_flattens_then_resumes():
    """The switch must fire on a deep loss, and must not latch on for ever."""
    idx = pd.date_range("2021-01-01", periods=120, freq="1D", tz="UTC")
    close = pd.Series(100.0, index=idx)
    close.iloc[20:40] = np.linspace(100.0, 40.0, 20)   # crash
    close.iloc[40:] = np.linspace(40.0, 90.0, 80)      # recovery
    df = pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995, "close": close, "volume": 1.0}
    )
    risk = RiskConfig(target_vol=0.0, atr_stop_mult=0.0, max_drawdown_stop=0.25, cooldown_bars=5)
    res = BacktestEngine(FRICTIONLESS, risk).run(df, pd.Series(1.0, index=idx))
    assert res.meta["kill_switch_events"] >= 1
    gross = res.weights.abs().sum(axis=1)
    assert (gross.iloc[-20:] > 0).any(), "agent never resumed trading after the cooldown"


def test_account_can_be_wiped_out_and_stays_dead():
    idx = pd.date_range("2021-01-01", periods=30, freq="1D", tz="UTC")
    close = pd.Series(100.0, index=idx)
    close.iloc[10] = 1.0          # -99% in one bar, held with leverage
    close.iloc[11:] = 1.0
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1.0}
    )
    cfg = ExecutionConfig(initial_capital=100.0, fee_bps=0.0, slippage_bps=0.0,
                          min_trade_frac=0.0, max_leverage=3.0)
    res = BacktestEngine(cfg, NO_RISK_LAYER).run(df, pd.Series(3.0, index=idx))
    assert res.meta["bust"] is True
    assert res.equity.iloc[-1] == 0.0
    assert (res.equity.iloc[12:] == 0.0).all()


def test_dust_trades_are_skipped(trending_prices):
    noisy = pd.Series(np.linspace(0.50, 0.55, len(trending_prices)), index=trending_prices.index)
    chatty = BacktestEngine(
        ExecutionConfig(initial_capital=100.0, min_trade_frac=0.0), NO_RISK_LAYER
    ).run(trending_prices, noisy)
    patient = BacktestEngine(
        ExecutionConfig(initial_capital=100.0, min_trade_frac=0.25), NO_RISK_LAYER
    ).run(trending_prices, noisy)
    assert len(patient.trades) < len(chatty.trades)


def test_multi_asset_shares_one_pot(prices):
    panel = {"a": prices, "b": prices * 1.0}
    w = pd.DataFrame({"a": 0.5, "b": 0.5}, index=prices.index)
    res = BacktestEngine(FRICTIONLESS, NO_RISK_LAYER).run(panel, w)
    assert list(res.weights.columns) == ["a", "b"]
    assert res.weights.abs().sum(axis=1).max() <= 1.05


def test_buy_and_hold_benchmark_matches_price_path(trending_prices):
    bh = buy_and_hold_equity(trending_prices, 100.0, fee_bps=0.0)
    ratio = bh.iloc[-1] / bh.iloc[0]
    price_ratio = trending_prices["close"].iloc[-1] / trending_prices["close"].iloc[0]
    assert ratio == pytest.approx(price_ratio, rel=1e-12)
