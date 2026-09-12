"""A compounding backtest framework for a small-account trading agent.

The package is organised as a small pipeline:

    data        -> OHLCV loading (Coinbase, Yahoo, CSV, synthetic)
    indicators  -> causal technical features
    strategies  -> features -> target weight in [-1, 1]
    agent       -> ensemble of strategies + regime filter + risk manager
    engine      -> execution, costs, leverage, stops, compounding
    optimize    -> walk-forward (out-of-sample) parameter selection
    metrics     -> performance statistics, incl. time-to-target
    report      -> plots and text summaries

Everything is causal: a strategy may only look at bars up to and including
bar ``t`` when it sets the weight it wants to hold *during* bar ``t + 1``.
"""

__version__ = "0.1.0"

from .engine import BacktestEngine, BacktestResult, ExecutionConfig
from .agent import TradingAgent, AgentConfig
from .metrics import summarize, time_to_target

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "ExecutionConfig",
    "TradingAgent",
    "AgentConfig",
    "summarize",
    "time_to_target",
]
