import numpy as np
import pandas as pd
import pytest

from tradingagent import metrics as m


@pytest.fixture
def doubling_equity():
    idx = pd.date_range("2020-01-01", periods=366, freq="1D", tz="UTC")
    return pd.Series(100.0 * (2.0 ** (np.arange(366) / 365.0)), index=idx)


def test_total_return_and_cagr(doubling_equity):
    assert m.total_return(doubling_equity) == pytest.approx(1.0, rel=1e-9)
    assert m.cagr(doubling_equity, 365.0) == pytest.approx(1.0, rel=1e-6)


def test_max_drawdown_matches_hand_calculation():
    idx = pd.date_range("2020-01-01", periods=5, freq="1D", tz="UTC")
    eq = pd.Series([100.0, 120.0, 60.0, 90.0, 110.0], index=idx)
    assert m.max_drawdown(eq) == pytest.approx(-0.5)


def test_sharpe_of_a_constant_series_is_zero():
    idx = pd.date_range("2020-01-01", periods=100, freq="1D", tz="UTC")
    assert m.sharpe(pd.Series(0.001, index=idx), 365.0) == 0.0


def test_time_to_target_finds_first_crossing():
    idx = pd.date_range("2020-01-01", periods=10, freq="1D", tz="UTC")
    eq = pd.Series([100, 200, 400, 900, 1000, 1500, 800, 1200, 1300, 1400], index=idx, dtype=float)
    out = m.time_to_target(eq, 1000.0, 365.0)
    assert out["target_hit"] == 1.0
    assert out["bars_to_target"] == 4
    assert out["date_to_target"] == idx[4]


def test_time_to_target_reports_a_miss():
    idx = pd.date_range("2020-01-01", periods=5, freq="1D", tz="UTC")
    out = m.time_to_target(pd.Series(100.0, index=idx), 1000.0)
    assert out["target_hit"] == 0.0
    assert np.isnan(out["bars_to_target"])


def test_monte_carlo_reports_probabilities():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=800, freq="1D", tz="UTC")
    rets = pd.Series(rng.normal(0.004, 0.03, 800), index=idx)
    out = m.monte_carlo_paths(rets, initial_capital=100.0, target=1000.0, n_paths=300, seed=1)
    assert 0.0 <= out["p_hit_target"] <= 1.0
    assert 0.0 <= out["p_ruin"] <= 1.0
    assert out["p05_final_equity"] <= out["median_final_equity"] <= out["p95_final_equity"]


def test_monte_carlo_detects_ruin_in_a_losing_edge():
    idx = pd.date_range("2020-01-01", periods=500, freq="1D", tz="UTC")
    rets = pd.Series(-0.02, index=idx)
    out = m.monte_carlo_paths(rets, initial_capital=100.0, target=1000.0, n_paths=50, seed=2)
    assert out["p_hit_target"] == 0.0
    assert out["p_ruin"] > 0.9


def test_deflated_sharpe_falls_as_the_search_widens():
    few = m.deflated_sharpe(1.5, n_trials=5, n_obs=1000)
    many = m.deflated_sharpe(1.5, n_trials=5000, n_obs=1000)
    assert few > many


def test_format_summary_mentions_the_target():
    idx = pd.date_range("2020-01-01", periods=3, freq="1D", tz="UTC")
    stats = {
        "start": idx[0], "end": idx[-1], "bars": 3.0, "initial_equity": 100.0,
        "final_equity": 1500.0, "total_return": 14.0, "cagr": 2.0, "ann_vol": 0.5,
        "max_drawdown": -0.2, "sharpe": 1.2, "sortino": 1.5, "calmar": 10.0,
        "ulcer_index": 3.0, "n_trades": 20, "turnover_per_year": 5.0,
        "time_in_market": 0.8, "avg_gross_exposure": 1.0, "total_costs": 3.0,
        "cost_drag": 0.03, "target_hit": 1.0, "bars_to_target": 2, "years_to_target": 0.01,
        "date_to_target": idx[2],
    }
    text = m.format_summary(stats, "Test")
    assert "TARGET REACHED" in text
    assert "$1,500.00" in text


def test_benchmark_is_rebased_to_the_same_start():
    """Comparing levels across different start dates is not a comparison."""
    idx = pd.date_range("2020-01-01", periods=5, freq="1D", tz="UTC")
    equity = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0], index=idx)
    # a benchmark curve that was already at 1000 when the account opened at 100
    bench = pd.Series([1000.0, 1100.0, 1200.0, 1300.0, 1400.0], index=idx)

    class R:
        exec_config = type("E", (), {"periods_per_year": 365.0, "initial_capital": 100.0,
                                     "target_equity": 1000.0})()
        returns = equity.pct_change().fillna(0.0)
        weights = pd.DataFrame(1.0, index=idx, columns=["a"])
        trades = pd.DataFrame()
        costs = pd.DataFrame(index=idx)
        meta = {}
    R.equity = equity

    stats = m.summarize(R(), benchmark=bench)
    # both grew 40% over the window, so both must finish at the same equity
    assert stats["benchmark_final_equity"] == pytest.approx(140.0)
    assert stats["benchmark_total_return"] == pytest.approx(0.4)
    assert stats["excess_return"] == pytest.approx(0.0, abs=1e-12)


def test_total_costs_means_the_same_thing_with_and_without_a_trade_log():
    """Regression: the field used to swallow financing when the log was off."""
    import numpy as np
    from tradingagent.engine import BacktestEngine, ExecutionConfig
    from tradingagent.risk import RiskConfig
    from tradingagent.data import synthetic_ohlcv

    prices = synthetic_ohlcv(300, seed=11)
    weights = pd.Series(1.8, index=prices.index)       # levered, so financing is non-zero
    risk = RiskConfig(target_vol=0.0, atr_stop_mult=0.0, max_drawdown_stop=0.0)

    logged = m.summarize(BacktestEngine(
        ExecutionConfig(initial_capital=100.0, max_leverage=2.0, min_trade_frac=0.0,
                        record_trades=True), risk).run(prices, weights))
    silent = m.summarize(BacktestEngine(
        ExecutionConfig(initial_capital=100.0, max_leverage=2.0, min_trade_frac=0.0,
                        record_trades=False), risk).run(prices, weights))

    assert logged["financing_paid"] > 0, "test needs a run that actually pays financing"
    assert silent["total_costs"] == pytest.approx(logged["total_costs"], rel=1e-9)
    assert silent["financing_paid"] == pytest.approx(logged["financing_paid"], rel=1e-9)
