"""What kind of number is this?

A Sharpe of 0.92 measured out-of-sample and a Sharpe of 0.92 measured on the
data the parameters were chosen from are not the same claim, and in this
repository they currently print identically. This module makes the difference
part of the result rather than something a reader is expected to remember.

The five kinds, in increasing order of what they are worth
----------------------------------------------------------
``IN_SAMPLE``
    Measured on data that influenced the parameters. Useful for sensitivity
    analysis and debugging, worthless as evidence of an edge.

``OUT_OF_SAMPLE``
    Walk-forward: parameters chosen on a training window, traded untouched on
    the window after it. Evidence, but weakened by however many configurations
    were tried - which is why ``n_configurations`` is required.

``HOLDOUT``
    A reserved era, spent the first time it is looked at. Strong evidence, and
    only the first look is worth anything; the ledger counts them.

``UNSEEN_ASSET``
    Frozen parameters on an instrument that touched nothing in their selection.
    The strongest evidence available without waiting, because there is no way
    to have fitted it.

``UNSEEN_PERIOD``
    Forward paper trading. The only test nothing in this repository can fool,
    and the slowest.

Pooling any two of these is the mistake this module exists to prevent, so
:func:`combine` refuses to average across kinds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import pandas as pd


class Sample(str, Enum):
    IN_SAMPLE = "IN-SAMPLE"
    OUT_OF_SAMPLE = "OUT-OF-SAMPLE"
    HOLDOUT = "HOLDOUT"
    UNSEEN_ASSET = "UNSEEN-ASSET"
    UNSEEN_PERIOD = "UNSEEN-PERIOD"

    @property
    def evidential_weight(self) -> int:
        """Rough ordering, for sorting a report so the weakest claims sit last."""
        return {
            Sample.IN_SAMPLE: 0,
            Sample.OUT_OF_SAMPLE: 1,
            Sample.HOLDOUT: 2,
            Sample.UNSEEN_ASSET: 3,
            Sample.UNSEEN_PERIOD: 4,
        }[self]


@dataclass
class Provenance:
    """Everything needed to know how much a number is worth.

    ``n_configurations`` is the count that a multiple-testing correction has to
    correct for. It is required rather than optional because a result quoted
    without it is not interpretable, and defaulting it to 1 would quietly assert
    that no search happened.
    """

    sample: Sample
    dataset: str
    start: Optional[str] = None
    end: Optional[str] = None
    n_configurations: int = 1
    n_datasets: int = 1
    n_validation_decisions: int = 0
    seeds: int = 1
    cost_scenario: str = "base"
    membership: str = "unknown"
    notes: str = ""

    def header(self) -> str:
        return (
            f"[{self.sample.value}] {self.dataset} {self.start or '?'} -> {self.end or '?'}  "
            f"| {self.n_configurations:,} configurations, {self.n_datasets} dataset(s), "
            f"{self.seeds} seed(s), {self.n_validation_decisions} validation decision(s) "
            f"| costs={self.cost_scenario} membership={self.membership}"
        )

    def row(self) -> Dict[str, object]:
        return {
            "sample": self.sample.value,
            "dataset": self.dataset,
            "start": self.start,
            "end": self.end,
            "n_configurations": self.n_configurations,
            "n_datasets": self.n_datasets,
            "n_validation_decisions": self.n_validation_decisions,
            "seeds": self.seeds,
            "cost_scenario": self.cost_scenario,
            "membership": self.membership,
            "notes": self.notes,
        }


@dataclass
class Claim:
    """A statistic plus the provenance that says what it is worth."""

    stats: Dict[str, float]
    provenance: Provenance

    def row(self) -> Dict[str, object]:
        out = dict(self.provenance.row())
        for key in ("final_equity", "total_return", "cagr", "sharpe", "sortino",
                    "max_drawdown", "n_trades", "turnover_per_year", "total_costs",
                    "avg_gross_exposure", "ann_vol"):
            if key in self.stats:
                out[key] = self.stats[key]
        return out


def table(claims: Sequence[Claim]) -> pd.DataFrame:
    """One row per claim, sorted strongest evidence first.

    The ``sample`` column is deliberately the first one: a reader scanning the
    table sees what kind of number it is before they see how big it is.
    """
    rows = [c.row() for c in claims]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    order = {s.value: s.evidential_weight for s in Sample}
    frame = frame.sort_values("sample", key=lambda c: c.map(order), ascending=False)
    first = [c for c in ("sample", "dataset", "start", "end") if c in frame.columns]
    return frame[first + [c for c in frame.columns if c not in first]]


def combine(claims: Sequence[Claim]) -> Dict[str, float]:
    """Refuses to average across sample kinds.

    Averaging an in-sample number with an unseen-asset number produces something
    that is not evidence of anything, and it is an easy mistake to make when a
    table has one row per asset. Raising is the point.
    """
    kinds = {c.provenance.sample for c in claims}
    if len(kinds) > 1:
        raise ValueError(
            "refusing to pool claims of different kinds: "
            f"{sorted(k.value for k in kinds)}. Report them separately."
        )
    if not claims:
        return {}
    keys = set().union(*(c.stats.keys() for c in claims))
    out: Dict[str, float] = {}
    for key in keys:
        values = [float(c.stats[key]) for c in claims
                  if key in c.stats and isinstance(c.stats[key], (int, float))]
        if values:
            out[key] = float(pd.Series(values).median())
    return out
