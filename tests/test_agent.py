import numpy as np
import pandas as pd
import pytest

from tradingagent.agent import AgentConfig, PortfolioAgent, TradingAgent
from tradingagent.engine import ExecutionConfig
from tradingagent.risk import RiskConfig


def test_blended_signal_is_causal(prices):
    """The adaptive weighting scores strategies on trailing P&L only."""
    agent = TradingAgent(AgentConfig())
    cut = 600
    full = agent.signal(prices).iloc[:cut]
    truncated = TradingAgent(AgentConfig()).signal(prices.iloc[:cut])
    pd.testing.assert_series_equal(full, truncated, check_names=False, rtol=1e-9, atol=1e-9)


def test_blend_weights_sum_to_one(prices):
    agent = TradingAgent(AgentConfig())
    agent.signal(prices)
    w = agent.diagnostics_["blend_weights"]
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-9)
    assert (w >= -1e-12).all().all()


def test_adaptive_weighting_favours_the_better_strategy(trending_prices):
    """In a clean uptrend the trend model must end up outweighing the fader."""
    agent = TradingAgent(AgentConfig(strategies=["ema_trend", "mean_reversion"], perf_lookback=60))
    agent.signal(trending_prices)
    w = agent.diagnostics_["blend_weights"].tail(50).mean()
    assert w["ema_trend"] > w["mean_reversion"]


def test_regime_filter_blocks_counter_trend_positions(prices):
    cfg = AgentConfig(regime_filter=True, regime_trend=100)
    agent = TradingAgent(cfg)
    sig = agent.signal(prices)
    from tradingagent import indicators as ind

    above = prices["close"] > ind.ema(prices["close"], 100)
    assert not ((sig < 0) & above).any()
    assert not ((sig > 0) & ~above).any()


def test_long_only_agent_never_shorts(prices):
    sig = TradingAgent(AgentConfig(allow_short=False)).signal(prices)
    assert (sig >= 0).all()


def test_signal_stays_inside_the_unit_band(prices):
    sig = TradingAgent(AgentConfig()).signal(prices)
    assert sig.between(-1.0, 1.0).all()


def test_risk_layer_respects_the_leverage_cap(prices):
    agent = TradingAgent(AgentConfig(), RiskConfig(target_vol=5.0, max_leverage=2.0))
    assert agent.sized_weight(prices).abs().max() <= 2.0 + 1e-12


def test_backtest_runs_end_to_end(prices):
    res = TradingAgent(AgentConfig()).backtest(prices, ExecutionConfig(initial_capital=100.0))
    assert len(res.equity) == len(prices)
    assert res.equity.iloc[0] > 0
    assert set(["sharpe", "max_drawdown", "final_equity"]).issubset(res.stats())


def test_portfolio_agent_allocates_across_symbols(prices):
    panel = {"a": prices, "b": prices.iloc[::-1].sort_index()}
    pa = PortfolioAgent(["a", "b"], AgentConfig(), RiskConfig(max_leverage=2.0))
    w = pa.target_weights(panel)
    assert list(w.columns) == ["a", "b"]
    res = pa.backtest(panel, ExecutionConfig(initial_capital=100.0, max_leverage=2.0))
    assert len(res.equity) == len(prices)
