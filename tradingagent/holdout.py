"""A held-out era you are supposed to look at once.

Walk-forward already keeps the optimiser out of the window it trades. It cannot
keep *you* out. Every time a researcher looks at a result, adjusts something and
looks again, the reported number drifts toward the data - and no statistic in
the pipeline can see that happening, because it happens in the researcher's head
between runs.

The only defence is a slice of history that is never used while building, and is
checked once at the end. This module makes that concrete in two ways:

* :class:`Holdout` **hides** the era. Code built against
  :meth:`Holdout.development` cannot accidentally read the reserved bars,
  because they are not in the frame it gets.
* It **counts**. Every evaluation against the reserved era is appended to a
  ledger on disk, with a timestamp and a label. A second look is not forbidden -
  sometimes you genuinely need one - but it is recorded, and the report says how
  many times the era has been used. A held-out era checked eleven times is not a
  held-out era, and the ledger is what stops that fact from quietly disappearing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .universe import Panel

class HoldoutViolation(RuntimeError):
    """Raised when reserved data reaches code that must not see it."""


DEFAULT_LEDGER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "holdout_ledger.json"
)

Data = Union[pd.DataFrame, Panel]


@dataclass
class Holdout:
    """Splits history into a development era and a reserved one.

    ``start`` is the first reserved bar. Everything before it is yours to
    iterate on; everything from it onward is spent the first time you look.
    """

    start: pd.Timestamp
    ledger_path: str = DEFAULT_LEDGER
    name: str = "default"

    def __post_init__(self):
        ts = pd.Timestamp(self.start)
        self.start = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    # -- splitting ------------------------------------------------------ #
    def development(self, data: Data) -> Data:
        """The part you may look at as often as you like."""
        if isinstance(data, Panel):
            return data.loc(end=self.start - pd.Timedelta(nanoseconds=1))
        return data.loc[: self.start - pd.Timedelta(nanoseconds=1)]

    def reserved(self, data: Data) -> Data:
        """The part that costs something to look at. Use :meth:`evaluate`."""
        if isinstance(data, Panel):
            return data.loc(start=self.start)
        return data.loc[self.start :]

    def covers(self, data: Data) -> bool:
        index = data.index if isinstance(data, (pd.DataFrame, Panel)) else data
        return bool(len(index)) and index[-1] >= self.start

    # -- protection ------------------------------------------------------ #
    def guard(self, data: Data, *, what: str = "this operation") -> Data:
        """Refuse to hand reserved bars to something that should not see them.

        Wrap the data going into an optimiser in this. If the data reaches into
        the reserved era the call fails loudly instead of quietly training on
        it - which is the failure mode that cannot be detected afterwards,
        because nothing about a leaked holdout looks wrong in the output.

        Returns the development split, so the honest call is also the easy one::

            wf = walk_forward(holdout.guard(panel), ...)
        """
        if not self.covers(data):
            return data
        index = data.index
        overlap = int((index >= self.start).sum())
        raise HoldoutViolation(
            f"{what} was handed {overlap:,} bars from the reserved era "
            f"(from {self.start.date()}). The holdout exists so that exactly this "
            f"cannot happen by accident.\n"
            f"  - to train on development data only:  holdout.development(data)\n"
            f"  - to spend the holdout deliberately:  holdout.evaluate(...)"
        )

    def protect(self, data: Data) -> Data:
        """``guard`` without the exception: silently returns the development split.

        Use when a caller legitimately has the full history and simply must not
        optimise over the end of it.
        """
        return self.development(data) if self.covers(data) else data

    # -- the ledger ----------------------------------------------------- #
    def _read_ledger(self) -> List[dict]:
        if not os.path.exists(self.ledger_path):
            return []
        try:
            with open(self.ledger_path) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):  # pragma: no cover - corrupt file
            return []

    def uses(self) -> List[dict]:
        """Every recorded look at this holdout, oldest first."""
        return [row for row in self._read_ledger() if row.get("name") == self.name]

    def n_uses(self) -> int:
        return len(self.uses())

    def record(self, label: str, stats: Optional[Dict[str, float]] = None) -> int:
        """Append a use to the ledger and return the new count."""
        ledger = self._read_ledger()
        entry = {
            "name": self.name,
            "label": label,
            "holdout_start": self.start.isoformat(),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if stats:
            entry["stats"] = {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                for k, v in stats.items()
                if k in ("final_equity", "sharpe", "max_drawdown", "total_return", "calmar")
            }
        ledger.append(entry)
        os.makedirs(os.path.dirname(os.path.abspath(self.ledger_path)), exist_ok=True)
        with open(self.ledger_path, "w") as fh:
            json.dump(ledger, fh, indent=1)
        return len([r for r in ledger if r.get("name") == self.name])

    # -- the one call that spends it ------------------------------------ #
    def evaluate(self, label: str, stats: Dict[str, float], *, verbose: bool = True) -> Dict[str, float]:
        """Record a look at the reserved era and annotate the result.

        Returns the stats with two fields added: how many times this era has now
        been used, and a plain-language reading of what that count means for how
        much the number should be trusted.
        """
        count = self.record(label, stats)
        out = dict(stats)
        out["holdout_uses"] = float(count)
        out["holdout_verdict"] = self._verdict(count)
        if verbose:
            print(self.report(latest=out))
        return out

    @staticmethod
    def _verdict(count: int) -> str:
        if count <= 1:
            return "first look - this is a genuine out-of-sample result"
        if count <= 3:
            return f"look #{count} - mild multiple-testing; treat the number as indicative"
        if count <= 10:
            return f"look #{count} - this era is no longer held out in any meaningful sense"
        return f"look #{count} - this is now a training set; reserve a fresh era"

    def report(self, latest: Optional[Dict[str, float]] = None) -> str:
        uses = self.uses()
        lines = [
            f"== holdout '{self.name}' - reserved from {self.start.date()} ==",
            f"  times evaluated   {len(uses)}",
        ]
        if latest:
            lines += [
                f"  final equity      ${latest.get('final_equity', float('nan')):,.2f}",
                f"  sharpe            {latest.get('sharpe', float('nan')):.2f}",
                f"  max drawdown      {latest.get('max_drawdown', float('nan')):.1%}",
                f"  verdict           {latest.get('holdout_verdict', '')}",
            ]
        if len(uses) > 1:
            lines.append("  previous looks:")
            for row in uses[:-1] if latest else uses:
                stats = row.get("stats", {})
                equity = stats.get("final_equity")
                shown = f"${equity:,.2f}" if isinstance(equity, float) else "n/a"
                lines.append(f"    {row['at'][:16]}  {row['label'][:36]:<36} {shown}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear this holdout's ledger entries - for tests, not for research."""
        ledger = [row for row in self._read_ledger() if row.get("name") != self.name]
        os.makedirs(os.path.dirname(os.path.abspath(self.ledger_path)), exist_ok=True)
        with open(self.ledger_path, "w") as fh:
            json.dump(ledger, fh, indent=1)
