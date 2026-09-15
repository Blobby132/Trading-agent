"""Does the mechanism generalise, or only this asset?

Every crypto conclusion in this repository rests on BTC-USD and every stock
conclusion on one 125-name universe. That is one draw from the population of
markets, and a strategy fitted to one draw will look fine on it whether or not
the mechanism is real.

The test here is deliberately unflattering: take the configuration the search
chose on the **trained** asset, **freeze it**, and run it unchanged elsewhere.
No re-selection, no per-asset tuning, no quiet substitution of a better
parameter set. Re-optimising per asset would answer a different and much easier
question - "can this framework fit any series?" - to which the answer is
obviously yes and is worth nothing.

Three tiers, and they must not be pooled
----------------------------------------
``trained``
    The asset the parameters were selected on. Its result is **in-sample with
    respect to parameter choice** even when the walk-forward made it
    out-of-sample in time. It is reported for reference, never as evidence.

``validation``
    A different instrument in the same asset class and the same broad regime
    (ETH against BTC). Related enough that the mechanism should transfer if it
    is real; correlated enough that passing is weak evidence.

``unseen``
    A different asset class entirely (US equity index ETFs). Nothing about
    these touched the parameter choice. This is the tier that carries the
    evidence, and it is the one most likely to fail.

A strategy that works on the trained asset, degrades on validation and fails on
unseen is showing the signature of a fit, not of a mechanism.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .baselines import baseline_table
from .engine import BacktestEngine, ExecutionConfig
from .metrics import summarize
from .optimize import DEFAULT_SEARCH_SPACE, WalkForwardResult, params_to_configs, sized_weight_for

#: Tier labels. Kept as constants because pooling these in a report is the
#: specific mistake this module exists to prevent.
TRAINED = "trained"
VALIDATION = "validation"
UNSEEN = "unseen"


@dataclass(frozen=True)
class FrozenConfig:
    """One parameter set, fixed, with a record of where it came from.

    ``source`` and ``chosen_in_folds`` exist so a result can never be described
    as unseen when the configuration behind it was picked on that very data.
    """

    params: Dict
    source: str
    chosen_in_folds: int = 0
    total_folds: int = 0

    def describe(self) -> str:
        return (
            f"frozen from {self.source}: chosen in {self.chosen_in_folds}/{self.total_folds} "
            f"folds ({self.chosen_in_folds / max(self.total_folds, 1):.0%})"
        )


def freeze_from_walk_forward(result: WalkForwardResult, source: str) -> FrozenConfig:
    """The configuration the walk-forward picked most often, frozen.

    A walk-forward re-selects every fold, so it does not hand back one
    parameter set. The modal top-1 choice is the honest summary: it is what the
    procedure converged on most often, and freezing anything else - the last
    fold's pick, or the best-performing fold's - would be choosing with
    hindsight.

    Ties break on the first occurrence, which makes the result deterministic.
    """
    if not result.chosen:
        raise ValueError("walk-forward selected nothing; cannot freeze a configuration")
    keys = [tuple(sorted(fold[0].items())) for fold in result.chosen]
    counts = Counter(keys)
    winner, n = counts.most_common(1)[0]
    return FrozenConfig(
        params=dict(winner), source=source,
        chosen_in_folds=int(n), total_folds=len(keys),
    )


def apply_frozen(
    config: FrozenConfig,
    data: pd.DataFrame,
    *,
    base_exec: Optional[ExecutionConfig] = None,
) -> Dict[str, float]:
    """Run a frozen configuration over one asset. No search of any kind.

    The weight series is still computed causally bar by bar - freezing the
    parameters does not make the signal non-causal - but nothing about this data
    influences which parameters are used.
    """
    base_exec = base_exec or ExecutionConfig()
    weights, exec_cfg, risk_cfg = sized_weight_for(data, config.params, base_exec)
    result = BacktestEngine(exec_cfg, risk_cfg).run(data, weights)
    return summarize(result)


@dataclass
class AssetSpec:
    """One asset in the study, with its tier and its calendar."""

    symbol: str
    tier: str
    source: str = "coinbase"
    periods_per_year: float = 365.0
    start: str = "2016-01-01"
    note: str = ""


def cross_asset_table(
    config: FrozenConfig,
    datasets: Dict[str, pd.DataFrame],
    specs: Dict[str, AssetSpec],
    *,
    initial_capital: float = 100.0,
    cost_model=None,
    with_baselines: bool = True,
) -> pd.DataFrame:
    """Frozen configuration against every asset, beside that asset's own baselines.

    An absolute return on an unseen asset says nothing on its own - a rising
    market lifts any long-biased rule. What matters is the strategy against
    *that asset's* buy-and-hold and simple-rule baselines over the identical
    bars, which is why the baselines are computed per asset rather than once.
    """
    rows: List[Dict[str, object]] = []
    for symbol, data in datasets.items():
        spec = specs[symbol]
        base = ExecutionConfig(
            initial_capital=initial_capital,
            periods_per_year=spec.periods_per_year,
            record_trades=False,
        )
        if cost_model is not None:
            base = replace(base, costs=cost_model)
        stats = apply_frozen(config, data, base_exec=base)
        row = {
            "symbol": symbol,
            "tier": spec.tier,
            "bars": stats.get("bars"),
            "start": data.index[0].date(),
            "end": data.index[-1].date(),
            "final_equity": stats.get("final_equity"),
            "cagr": stats.get("cagr"),
            "sharpe": stats.get("sharpe"),
            "sortino": stats.get("sortino"),
            "max_drawdown": stats.get("max_drawdown"),
            "n_trades": stats.get("n_trades"),
            "time_in_market": stats.get("time_in_market"),
            "total_costs": stats.get("total_costs"),
        }
        if with_baselines:
            bl = baseline_table(data, base_exec=base).set_index("baseline")
            for name in ("buy_and_hold", "price_above_sma_200", "momentum_12m"):
                if name in bl.index:
                    row[f"bl_{name}"] = float(bl.loc[name, "final_equity"])
                    row[f"bl_{name}_sharpe"] = float(bl.loc[name, "sharpe"])
            if "buy_and_hold" in bl.index:
                bh = float(bl.loc["buy_and_hold", "final_equity"])
                row["excess_vs_buy_hold"] = row["final_equity"] - bh
                row["beats_buy_hold"] = bool(row["final_equity"] > bh)
            if "price_above_sma_200" in bl.index:
                sma = float(bl.loc["price_above_sma_200", "final_equity"])
                row["beats_sma_200"] = bool(row["final_equity"] > sma)
        rows.append(row)
    table = pd.DataFrame(rows)
    order = {TRAINED: 0, VALIDATION: 1, UNSEEN: 2}
    return table.sort_values(["tier", "symbol"], key=lambda c: c.map(order) if c.name == "tier" else c)


def verdict(table: pd.DataFrame) -> Dict[str, object]:
    """A blunt summary of whether the mechanism transferred.

    Reported per tier, never pooled: an average over trained and unseen assets
    would let the asset the parameters were fitted on carry the ones they were
    not.
    """
    out: Dict[str, object] = {}
    for tier in (TRAINED, VALIDATION, UNSEEN):
        sub = table[table["tier"] == tier]
        if sub.empty:
            continue
        out[tier] = {
            "assets": int(len(sub)),
            "median_sharpe": float(sub["sharpe"].median()),
            "median_final_equity": float(sub["final_equity"].median()),
            "beats_buy_hold": int(sub.get("beats_buy_hold", pd.Series(dtype=bool)).sum()),
            "beats_sma_200": int(sub.get("beats_sma_200", pd.Series(dtype=bool)).sum()),
            "worst_sharpe": float(sub["sharpe"].min()),
            "worst_drawdown": float(sub["max_drawdown"].min()),
        }
    return out
