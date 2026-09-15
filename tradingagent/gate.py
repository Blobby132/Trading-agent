"""The acceptance gate: what a strategy has to clear to be believed.

A strategy improvement must not be accepted because total return went up, or
CAGR went up, or Sharpe went up, or the drawdown got shallower, or the equity
curve looks better. Every one of those can be produced by searching, and this
repository has produced all of them by searching at least once.

So the gate is multi-dimensional and it is **default-deny**. Three properties
make it hard to fool:

1. **Silence is failure.** A check that was never run counts as failed, not as
   passed or as unknown. The common way a framework like this gets subverted is
   not by lying about a result, it is by not running the test that would have
   produced one.
2. **The comparison is a matched baseline, not zero.** Beating cash is not an
   achievement. The benchmark check asks whether the strategy beats the
   simplest alternative *with the same risk posture* - for a strategy that is
   half invested on average, that is a constant half-sized position, not full
   buy-and-hold.
3. **Rejection is a successful outcome.** "No improvement accepted" is a
   result. It is the result whenever the evidence does not support one, and a
   gate that never returns it is decoration.

The floors are deliberately modest. They are not targets to optimise toward -
optimising against this gate would defeat it - they are the level below which a
result is not worth acting on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class Check:
    """One dimension of the gate.

    ``passed=None`` means the evidence was never collected, which the gate
    treats as a failure.
    """

    name: str
    question: str
    passed: Optional[bool] = None
    detail: str = "not evaluated"
    required: bool = True

    @property
    def status(self) -> str:
        if self.passed is None:
            return "NOT RUN"
        return "PASS" if self.passed else "FAIL"

    @property
    def blocking(self) -> bool:
        """A required check that did not positively pass blocks acceptance."""
        return self.required and self.passed is not True


@dataclass
class GateResult:
    checks: List[Check] = field(default_factory=list)
    subject: str = "strategy"

    @property
    def accepted(self) -> bool:
        return not any(c.blocking for c in self.checks)

    @property
    def blockers(self) -> List[Check]:
        return [c for c in self.checks if c.blocking]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"check": c.name, "status": c.status, "required": c.required,
             "question": c.question, "detail": c.detail}
            for c in self.checks
        ])

    def report(self) -> str:
        lines = [f"== acceptance gate: {self.subject} ==", ""]
        for c in self.checks:
            mark = {"PASS": "PASS", "FAIL": "FAIL", "NOT RUN": "----"}[c.status]
            flag = "" if c.required else "  (advisory)"
            lines.append(f"  [{mark}] {c.name}{flag}")
            lines.append(f"         {c.detail}")
        lines.append("")
        if self.accepted:
            lines.append("  VERDICT: ACCEPTED - every required dimension cleared.")
        else:
            lines.append(f"  VERDICT: NOT ACCEPTED - {len(self.blockers)} blocking check(s):")
            for c in self.blockers:
                lines.append(f"    - {c.name}: {c.detail}")
            lines.append("")
            lines.append("  This is a successful outcome for the gate. It means the evidence")
            lines.append("  does not support the change, not that the test went wrong.")
        return "\n".join(lines)


#: Floors. Modest on purpose - see the module docstring.
DEFAULT_FLOORS = {
    "min_oos_sharpe": 0.30,
    "max_drawdown": -0.60,
    "min_trades": 30,
    "min_stability": 0.50,
    "min_deflated_sharpe": 0.50,      # a coin scores 0.50; below it is not evidence
    "max_null_p_value": 0.10,
    "min_regimes_positive_excess": 2,  # must add value in more than one regime
    "min_unseen_beating_benchmark": 0.5,  # at least half the unseen assets
    "max_participation": 0.01,         # 1% of daily volume
    "min_excess_over_matched_baseline": 0.0,
}


def evaluate(
    *,
    oos_stats: Optional[Dict[str, float]] = None,
    matched_baseline_stats: Optional[Dict[str, float]] = None,
    stability: Optional[pd.DataFrame] = None,
    cost_ladder: Optional[pd.DataFrame] = None,
    regime_report: Optional[pd.DataFrame] = None,
    cross_asset: Optional[pd.DataFrame] = None,
    deflated_sharpe: Optional[float] = None,
    null_comparison: Optional[pd.DataFrame] = None,
    capacity: Optional[Dict[str, float]] = None,
    parity_tests_pass: Optional[bool] = None,
    holdout_untouched: Optional[bool] = None,
    floors: Optional[Dict[str, float]] = None,
    subject: str = "strategy",
) -> GateResult:
    """Run every dimension. Anything not supplied is a failure, not a pass."""
    f = {**DEFAULT_FLOORS, **(floors or {})}
    checks: List[Check] = []

    # 1. out-of-sample performance
    c = Check("oos_performance", "Does it clear the floors out of sample?")
    if oos_stats:
        sharpe = float(oos_stats.get("sharpe", float("nan")))
        dd = float(oos_stats.get("max_drawdown", float("nan")))
        trades = float(oos_stats.get("n_trades", 0))
        fails = []
        if not np.isfinite(sharpe) or sharpe < f["min_oos_sharpe"]:
            fails.append(f"sharpe {sharpe:.2f} < {f['min_oos_sharpe']}")
        if not np.isfinite(dd) or dd < f["max_drawdown"]:
            fails.append(f"drawdown {dd:.1%} worse than {f['max_drawdown']:.0%}")
        if trades < f["min_trades"]:
            fails.append(f"only {int(trades)} trades")
        c.passed = not fails
        c.detail = "; ".join(fails) if fails else (
            f"sharpe {sharpe:.2f}, drawdown {dd:.1%}, {int(trades)} trades"
        )
    checks.append(c)

    # 2. beats a matched-risk simple baseline
    c = Check("beats_matched_baseline",
              "Does it beat the simplest alternative at the same risk posture?")
    if oos_stats and matched_baseline_stats:
        s_sharpe = float(oos_stats.get("sharpe", float("nan")))
        b_sharpe = float(matched_baseline_stats.get("sharpe", float("nan")))
        s_sortino = float(oos_stats.get("sortino", float("nan")))
        b_sortino = float(matched_baseline_stats.get("sortino", float("nan")))
        margin = s_sharpe - b_sharpe
        c.passed = bool(margin > f["min_excess_over_matched_baseline"] and s_sortino >= b_sortino)
        c.detail = (
            f"sharpe {s_sharpe:.3f} vs baseline {b_sharpe:.3f} ({margin:+.3f}); "
            f"sortino {s_sortino:.3f} vs {b_sortino:.3f}"
        )
    checks.append(c)

    # 3. parameter stability
    c = Check("parameter_stability", "Does it survive small parameter moves?")
    if stability is not None and not stability.empty:
        median = float(stability["stability"].median())
        spikes = stability[stability["verdict"].astype(str).str.startswith("SPIKE")]
        c.passed = bool(median >= f["min_stability"])
        c.detail = (f"median stability {median:.3f}, {len(spikes)} spike(s)"
                    + (f": {list(spikes['parameter'])}" if len(spikes) else ""))
    checks.append(c)

    # 4. cost robustness
    c = Check("cost_robustness", "Does it survive materially worse execution?")
    if cost_ladder is not None and not cost_ladder.empty and matched_baseline_stats:
        target = float(matched_baseline_stats.get("final_equity", float("inf")))
        ok = cost_ladder[cost_ladder["final_equity"] > target]
        breakeven = float(ok["bps_per_side"].max()) if len(ok) else float("nan")
        c.passed = bool(np.isfinite(breakeven) and breakeven >= 35.0)   # the HIGH scenario
        c.detail = f"beats the matched baseline up to ~{breakeven:.0f} bps/side"
    checks.append(c)

    # 5. regime robustness
    c = Check("regime_robustness", "Does it add value in more than one regime?")
    if regime_report is not None and not regime_report.empty and "excess" in regime_report.columns:
        trend = regime_report[regime_report.get("dimension", "") == "trend"]
        positive = trend[trend["excess"] > 0]
        c.passed = bool(len(positive) >= f["min_regimes_positive_excess"])
        c.detail = (f"positive excess in {len(positive)}/{len(trend)} trend regimes"
                    + (f": {list(positive['regime'])}" if len(positive) else ""))
    checks.append(c)

    # 6. cross-asset
    c = Check("cross_asset", "Do frozen parameters work on assets they never saw?")
    if cross_asset is not None and not cross_asset.empty:
        unseen = cross_asset[cross_asset["tier"] == "unseen"]
        if len(unseen):
            beat = int(unseen.get("beats_buy_hold", pd.Series(dtype=bool)).sum())
            share = beat / len(unseen)
            c.passed = bool(share >= f["min_unseen_beating_benchmark"])
            c.detail = f"{beat}/{len(unseen)} unseen assets beat buy-and-hold ({share:.0%})"
        else:
            c.detail = "no unseen assets tested"
    checks.append(c)

    # 7. statistical evidence
    # Two pieces of evidence, and the check distinguishes "neither was collected"
    # (NOT RUN, like every other dimension) from "collected and insufficient"
    # (FAIL). Both block acceptance; reporting them as the same thing would hide
    # which of the two problems the researcher actually has.
    c = Check("statistical_evidence", "Is it distinguishable from luck and from noise?")
    have_dsr = deflated_sharpe is not None and np.isfinite(deflated_sharpe)
    have_null = null_comparison is not None and not null_comparison.empty
    if have_dsr or have_null:
        parts, ok = [], True
        if have_dsr:
            parts.append(f"deflated Sharpe {deflated_sharpe:.3f}")
            ok &= deflated_sharpe >= f["min_deflated_sharpe"]
        else:
            ok = False
            parts.append("deflated Sharpe not supplied")
        if have_null:
            worst_p = float(null_comparison["p_value"].max())
            parts.append(f"worst null p-value {worst_p:.3f}")
            ok &= worst_p <= f["max_null_p_value"]
        else:
            ok = False
            parts.append("null comparison not supplied")
        c.passed = bool(ok)
        c.detail = "; ".join(parts)
    checks.append(c)

    # 8. turnover / capacity
    c = Check("capacity", "Do the returns depend on unrealistic volume?")
    if capacity:
        part = float(capacity.get("max_participation", float("nan")))
        c.passed = bool(np.isfinite(part) and part <= f["max_participation"])
        c.detail = f"max participation {part:.2e} of daily volume"
    checks.append(c)

    # 9. execution realism
    c = Check("execution_realism", "Does paper trading reproduce the backtest exactly?")
    if parity_tests_pass is not None:
        c.passed = bool(parity_tests_pass)
        c.detail = "backtest/paper parity tests pass" if parity_tests_pass else "parity tests FAIL"
    checks.append(c)

    # 10. holdout discipline - advisory, because a spent holdout is a fact about
    # the research process rather than about the strategy
    c = Check("holdout_discipline", "Has the reserved era been kept reserved?", required=False)
    if holdout_untouched is not None:
        c.passed = bool(holdout_untouched)
        c.detail = "holdout untouched" if holdout_untouched else "holdout has been evaluated against"
    checks.append(c)

    return GateResult(checks=checks, subject=subject)
