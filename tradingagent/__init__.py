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
from .agent import AgentConfig, PortfolioAgent, TradingAgent
from .risk import RiskConfig
from .metrics import monte_carlo_paths, summarize, time_to_target
from .optimize import (
    WalkForwardConfig,
    WalkForwardResult,
    search_until_target,
    walk_forward,
)

__all__ = [
    "AgentConfig",
    "BacktestEngine",
    "BacktestResult",
    "ExecutionConfig",
    "PortfolioAgent",
    "RiskConfig",
    "TradingAgent",
    "WalkForwardConfig",
    "WalkForwardResult",
    "monte_carlo_paths",
    "search_until_target",
    "summarize",
    "time_to_target",
    "walk_forward",
]
