"""Simple alternatives the agent has to beat to justify itself.

The question this module exists to answer is not "does the agent make money?"
It is: **does the agent's complexity buy anything a one-line rule would not?**

The agent is an ensemble of six strategies with adaptive performance weighting,
a regime filter, volatility targeting, ATR stops, a drawdown kill switch and a
walk-forward search over thousands of configurations. If a 200-day moving
average crossover, traded through the identical engine with the identical costs
over the identical window, gets to roughly the same place, then almost all of
that machinery is decoration - and worse, it is decoration with hundreds of
degrees of freedom attached to it.

Every baseline here:

* has **no free parameters that were searched** - the values are the textbook
  ones (50/200, 12-month lookback), fixed before looking at any result;
* runs through the **same** :class:`~tradingagent.engine.BacktestEngine` with
  the same :class:`~tradingagent.execution.CostModel`;
* is evaluated over the **same bars** as whatever it is being compared with.

They are deliberately not risk-managed. Adding vol targeting or stops to a
baseline would be tuning the baseline, and a baseline you tune is no longer a
baseline - it is another candidate strategy with the comparison built in.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from . import indicators as ind
from .engine import BacktestEngine, ExecutionConfig
from .metrics import summarize
from .risk import RiskConfig


# --------------------------------------------------------------------------- #
# the rules
# --------------------------------------------------------------------------- #
def cash(df: pd.DataFrame) -> pd.Series:
    """Hold nothing. The floor any strategy must clear to be worth running."""
    return pd.Series(0.0, index=df.index)


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Fully invested, always. One trade."""
    return pd.Series(1.0, index=df.index)


def sma_crossover(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    """Long while the fast average is above the slow one, flat otherwise.

    The textbook 50/200. Long-only, so it is runnable in a cash account and
    comparable to the long-only agent configurations.
    """
    c = df["close"]
    return (ind.sma(c, fast) > ind.sma(c, slow)).astype(float).fillna(0.0)


def price_above_sma(df: pd.DataFrame, n: int = 200) -> pd.Series:
    """Long while price is above its own 200-bar average. The simplest filter there is."""
    c = df["close"]
    return (c > ind.sma(c, n)).astype(float).fillna(0.0)


def simple_momentum(df: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """Long while the trailing 12-month return is positive.

    Time-series momentum as Moskowitz-Ooi-Pedersen state it, with no sizing, no
    smoothing and no volatility scaling.
    """
    c = df["close"]
    return (c.pct_change(lookback) > 0).astype(float).fillna(0.0)


#: Baselines keyed by name. Values are callables taking an OHLCV frame and
#: returning a target weight, on the same convention as
#: :class:`~tradingagent.strategies.Strategy`: the value at bar t is the
#: exposure wanted for bar t+1, and the engine does the shifting.
BASELINES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "cash": cash,
    "buy_and_hold": buy_and_hold,
    "sma_50_200": sma_crossover,
    "price_above_sma_200": price_above_sma,
    "momentum_12m": simple_momentum,
}


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def run_baseline(
    name: str,
    data: pd.DataFrame,
    *,
    base_exec: Optional[ExecutionConfig] = None,
    risk: Optional[RiskConfig] = None,
) -> Dict[str, float]:
    """One baseline through the real engine, with the real costs."""
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; known: {sorted(BASELINES)}")
    exec_cfg = base_exec or ExecutionConfig()
    # Risk layers off: a baseline is meant to be simple, and bolting the agent's
    # sizing onto it would make the comparison about sizing rather than signal.
    risk_cfg = risk or RiskConfig(
        target_vol=0.0, atr_stop_mult=0.0, max_drawdown_stop=0.0, reentry_lockout_bars=0
    )
    weights = BASELINES[name](data)
    result = BacktestEngine(exec_cfg, risk_cfg).run(data, weights)
    stats = summarize(result)
    stats["baseline"] = name
    return stats


def baseline_table(
    data: pd.DataFrame,
    *,
    base_exec: Optional[ExecutionConfig] = None,
    names: Sequence[str] = tuple(BASELINES),
    label: str = "",
) -> pd.DataFrame:
    """Every baseline over one dataset, ready to sit beside a strategy result."""
    rows = []
    for name in names:
        stats = run_baseline(name, data, base_exec=base_exec)
        rows.append({
            "baseline": name,
            "label": label,
            "final_equity": stats.get("final_equity"),
            "total_return": stats.get("total_return"),
            "cagr": stats.get("cagr"),
            "sharpe": stats.get("sharpe"),
            "sortino": stats.get("sortino"),
            "max_drawdown": stats.get("max_drawdown"),
            "calmar": stats.get("calmar"),
            "n_trades": stats.get("n_trades"),
            "total_costs": stats.get("total_costs"),
            "time_in_market": stats.get("time_in_market"),
            "bars": stats.get("bars"),
        })
    return pd.DataFrame(rows)


def excess_over_baselines(
    strategy_stats: Dict[str, float], baselines: pd.DataFrame, *, metric: str = "final_equity"
) -> pd.DataFrame:
    """How far ahead of each baseline the strategy finished, and whether it is ahead at all.

    ``beats`` is the column that matters. A strategy that does not beat
    ``price_above_sma_200`` has not earned the search that produced it.
    """
    value = float(strategy_stats.get(metric, float("nan")))
    out = baselines[["baseline", metric]].copy()
    out["strategy"] = value
    out["difference"] = value - out[metric]
    out["ratio"] = value / out[metric].replace(0.0, np.nan)
    out["beats"] = out["difference"] > 0
    return out
