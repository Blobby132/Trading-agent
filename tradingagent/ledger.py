"""A research ledger: every experiment, recorded, whether you like it or not.

Walk-forward keeps the optimiser out of the window it trades. Purging and
embargoes keep the target out of the features. Neither can keep *the
researcher* out: you look at a result, change something, and look again, and the
reported number drifts toward the data. That process leaves no trace in the code
and no trace in the statistics — which is precisely why the twentieth
configuration tried can look like the first.

This module makes it leave a trace. Every backtest that is meant to inform a
decision writes one row: what was run, over what, with which parameters and
seed, what came out, and — the field that matters most — whether the result was
used to *select* anything.

It is deliberately not clever. It is an append-only JSON file. The value is not
in the analytics; it is in the count being visible when you come to write down
what you found.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_LEDGER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "research_ledger.jsonl"
)

#: The statistics every entry records, so rows are comparable across experiments.
TRACKED_STATS = (
    "total_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown",
    "ann_vol", "turnover_per_year", "n_trades", "total_costs", "time_in_market",
    "avg_gross_exposure", "final_equity",
)


@dataclass
class Experiment:
    """One recorded run."""

    experiment_id: str
    timestamp: str
    label: str
    dataset_start: str
    dataset_end: str
    universe: str
    n_symbols: int
    candidate_count: int
    parameters: Dict[str, Any]
    seed: Optional[int]
    train_period: Optional[str]
    validation_period: Optional[str]
    holdout_status: str          # "development" | "validation" | "holdout" | "paper"
    objective: Optional[str]
    cost_scenario: str
    used_for_selection: bool
    stats: Dict[str, float]
    notes: str = ""

    def row(self) -> Dict[str, Any]:
        return asdict(self)


class ResearchLedger:
    """Append-only record of experiments, stored as JSON lines."""

    def __init__(self, path: str = DEFAULT_LEDGER):
        self.path = path

    # -- writing --------------------------------------------------------- #
    def record(
        self,
        label: str,
        stats: Dict[str, float],
        *,
        universe: str = "unknown",
        n_symbols: int = 0,
        dataset_start: Any = None,
        dataset_end: Any = None,
        candidate_count: int = 1,
        parameters: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        train_period: Optional[str] = None,
        validation_period: Optional[str] = None,
        holdout_status: str = "development",
        objective: Optional[str] = None,
        cost_scenario: str = "base",
        used_for_selection: bool = False,
        notes: str = "",
    ) -> Experiment:
        """Write one row. Returns the entry, including its generated id."""
        entry = Experiment(
            experiment_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            label=label,
            dataset_start=str(dataset_start)[:10] if dataset_start is not None else "",
            dataset_end=str(dataset_end)[:10] if dataset_end is not None else "",
            universe=universe,
            n_symbols=int(n_symbols),
            candidate_count=int(candidate_count),
            parameters=_jsonable(parameters or {}),
            seed=seed,
            train_period=train_period,
            validation_period=validation_period,
            holdout_status=holdout_status,
            objective=objective,
            cost_scenario=cost_scenario,
            used_for_selection=bool(used_for_selection),
            stats={
                k: float(stats[k])
                for k in TRACKED_STATS
                if k in stats and isinstance(stats[k], (int, float, np.floating))
                and np.isfinite(stats[k])
            },
            notes=notes,
        )
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(entry.row()) + "\n")
        return entry

    # -- reading --------------------------------------------------------- #
    def entries(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        rows = []
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:      # pragma: no cover - corrupt line
                    continue
        return rows

    def frame(self) -> pd.DataFrame:
        rows = self.entries()
        if not rows:
            return pd.DataFrame()
        flat = []
        for row in rows:
            item = {k: v for k, v in row.items() if k not in ("stats", "parameters")}
            item.update(row.get("stats", {}))
            item["n_parameters"] = len(row.get("parameters", {}))
            flat.append(item)
        return pd.DataFrame(flat)

    # -- the number that matters ----------------------------------------- #
    def selection_count(self, universe: Optional[str] = None) -> int:
        """How many results have been used to choose something.

        This is the effective number of trials for a multiple-testing
        correction. Feed it to :func:`~tradingagent.metrics.deflated_sharpe`
        rather than the count from a single search - the searches you ran last
        week still happened.
        """
        rows = [r for r in self.entries() if r.get("used_for_selection")]
        if universe is not None:
            rows = [r for r in rows if r.get("universe") == universe]
        return sum(int(r.get("candidate_count", 1)) for r in rows)

    def summary(self, universe: Optional[str] = None) -> str:
        rows = self.entries()
        if universe is not None:
            rows = [r for r in rows if r.get("universe") == universe]
        if not rows:
            return "== research ledger ==\n  empty"

        selecting = [r for r in rows if r.get("used_for_selection")]
        by_status: Dict[str, int] = {}
        for row in rows:
            by_status[row.get("holdout_status", "?")] = by_status.get(
                row.get("holdout_status", "?"), 0
            ) + 1
        trials = self.selection_count(universe)
        lines = [
            "== research ledger ==",
            f"  experiments recorded   {len(rows)}",
            f"  used for selection     {len(selecting)}",
            f"  effective trials       {trials:,}  (candidates summed over selecting runs)",
            f"  by stage               {by_status}",
            f"  first                  {rows[0]['timestamp'][:16]}",
            f"  latest                 {rows[-1]['timestamp'][:16]}",
        ]
        if trials > 200:
            lines.append(
                "  NOTE: with this many trials behind it, a headline Sharpe needs the "
                "deflated\n        version quoted beside it, using this count and not a "
                "single search's."
            )
        return "\n".join(lines)

    def reset(self) -> None:
        """Delete the ledger. For tests; not for research."""
        if os.path.exists(self.path):
            os.remove(self.path)


def _jsonable(obj: Any) -> Any:
    """Coerce parameters into something json can store."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
