"""Robustness testing: is the edge real, or did the parameters get lucky?

Three separate questions, three tools:

* :func:`parameter_stability` — does performance survive moving a parameter
  slightly? A configuration that works at lookback 50 and collapses at 49 was
  fitted to noise. A broad plateau is worth more than a taller spike.
* :func:`regime_report` — where does it work, where does it fail, and how badly?
  A strategy judged on one favourable era has not been judged.
* :func:`robustness_battery` — the whole gauntlet: costs, slippage, spreads,
  seeds, start dates, end dates, execution assumptions.

None of these tries to make a number bigger. They try to find the conditions
under which it goes away, which is the only way to learn whether it was ever
there.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .execution import COST_SCENARIOS, CostModel, cost_scenario

#: What a result has to clear to be taken seriously. Deliberately modest: these
#: are not targets to optimise toward, they are floors below which a strategy is
#: not worth paper-trading.
DEFAULT_ACCEPTANCE = {
    "min_sharpe": 0.3,
    "max_drawdown": -0.40,        # no worse than this
    "min_trades": 20,
    "min_stability": 0.50,        # parameter plateau, see parameter_stability
}


# --------------------------------------------------------------------------- #
# Phase 6 — parameter stability
# --------------------------------------------------------------------------- #
def neighbourhood(value, *, relative: float = 0.25, n_points: int = 5) -> List:
    """Nearby values for a parameter, inferred from its type.

    Integers get integer neighbours, floats get proportional ones, booleans and
    categoricals get both/all alternatives. Generic on purpose: hard-coding a
    grid for one parameter tells you nothing about the others.
    """
    if isinstance(value, bool):
        return [False, True]
    if isinstance(value, (int, np.integer)):
        span = max(int(round(abs(value) * relative)), 1)
        offsets = np.linspace(-span, span, n_points).round().astype(int)
        return sorted({int(value) + int(o) for o in offsets if int(value) + int(o) > 0})
    if isinstance(value, (float, np.floating)):
        if value == 0:
            return [0.0]
        factors = np.linspace(1 - relative, 1 + relative, n_points)
        return sorted({round(float(value) * f, 6) for f in factors})
    return [value]


def parameter_stability(
    evaluate: Callable[[Dict], Dict[str, float]],
    selected: Dict,
    *,
    metric: str = "sharpe",
    grids: Optional[Dict[str, Sequence]] = None,
    relative: float = 0.25,
    n_points: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-evaluate a chosen configuration at nearby parameter values.

    ``evaluate`` takes a parameter dict and returns a stats dict. Every
    parameter is perturbed one at a time, holding the rest at their selected
    values, and the spread of outcomes is reported.

    The **stability score** is the worst nearby result divided by the selected
    one, clipped to ``[0, 1]``. Near 1 means the neighbourhood is a plateau and
    the exact value did not matter. Near 0 means the selection sits on a spike,
    and the honest reading is that the parameter was fitted to this sample.
    """
    base = evaluate(dict(selected))
    base_score = float(base.get(metric, np.nan))
    rows = []

    for name, value in selected.items():
        candidates = list(grids[name]) if grids and name in grids else neighbourhood(
            value, relative=relative, n_points=n_points
        )
        candidates = [c for c in candidates if c != value]
        if not candidates:
            continue

        scores = []
        for candidate in candidates:
            trial = dict(selected)
            trial[name] = candidate
            try:
                scores.append(float(evaluate(trial).get(metric, np.nan)))
            except Exception:                       # noqa: BLE001 - a bad cell is data
                scores.append(float("nan"))

        finite = [s for s in scores if np.isfinite(s)]
        if not finite:
            continue
        worst, best, median = min(finite), max(finite), float(np.median(finite))
        if np.isfinite(base_score) and abs(base_score) > 1e-9 and base_score > 0:
            stability = float(np.clip(worst / base_score, 0.0, 1.0))
        else:
            stability = float("nan")

        rows.append({
            "parameter": name,
            "selected": value,
            "nearby": f"{min(candidates)} .. {max(candidates)}",
            "n_tested": len(candidates),
            "selected_score": base_score,
            "median": median,
            "worst": worst,
            "best": best,
            "stability": stability,
            "verdict": _stability_verdict(stability, base_score, best),
        })
        if verbose and rows:
            row = rows[-1]
            print(
                f"  {name:<20} {str(value):>8} -> median {median:>7.3f}  "
                f"worst {worst:>7.3f}  best {best:>7.3f}  stability {stability:>5.2f}  "
                f"{row['verdict']}"
            )

    frame = pd.DataFrame(rows)
    return frame.sort_values("stability") if not frame.empty else frame


def _stability_verdict(stability: float, selected: float, best: float) -> str:
    """Read the neighbourhood.

    The "a neighbour beats it" case is checked *first* and independently of the
    stability score, because those two findings are not the same thing. A
    selection whose neighbours all do better scores a perfect stability of 1.0 -
    nothing collapsed - while actually telling you the selection process picked
    the wrong cell. Ordering the checks the other way hid that entirely.
    """
    if np.isfinite(best) and np.isfinite(selected) and selected > 0 and best > selected * 1.2:
        return "a neighbour beats it by >20% - the selection looks arbitrary"
    if not np.isfinite(stability):
        return "n/a (selected score is not positive)"
    if stability >= 0.8:
        return "plateau"
    if stability >= 0.5:
        return "acceptable"
    return "SPIKE - performance collapses nearby; likely fitted to noise"


# --------------------------------------------------------------------------- #
# Phase 7 — market regimes
# --------------------------------------------------------------------------- #
def classify_regimes(
    prices: pd.Series,
    *,
    trend_lookback: int = 200,
    vol_lookback: int = 60,
    bull_threshold: float = 0.10,
    bear_threshold: float = -0.10,
    periods_per_year: float = 252.0,
) -> pd.DataFrame:
    """Label each bar by trend and volatility state.

    Labels are **causal** — computed from trailing windows and a trailing
    volatility percentile — so they describe a state that was knowable at the
    time. That matters less for reporting than it would for trading, but a
    label built from full-sample quantiles would quietly become look-ahead the
    moment somebody used it as a filter, and somebody always does.
    """
    trailing = prices.pct_change(trend_lookback)
    returns = np.log(prices).diff()
    vol = returns.rolling(vol_lookback, min_periods=vol_lookback // 2).std(ddof=0) * np.sqrt(
        periods_per_year
    )
    vol_rank = vol.rolling(4 * vol_lookback, min_periods=vol_lookback).rank(pct=True)

    trend = pd.Series("sideways", index=prices.index, dtype=object)
    trend[trailing >= bull_threshold] = "bull"
    trend[trailing <= bear_threshold] = "bear"
    trend[trailing.isna()] = "warmup"

    volatility = pd.Series("normal_vol", index=prices.index, dtype=object)
    volatility[vol_rank >= 0.8] = "high_vol"
    volatility[vol_rank <= 0.2] = "low_vol"
    volatility[vol_rank.isna()] = "warmup"

    drawdown = prices / prices.cummax() - 1.0
    crash = drawdown <= -0.20

    return pd.DataFrame({
        "trend": trend,
        "volatility": volatility,
        "in_crash": crash,
        "trailing_return": trailing,
        "realised_vol": vol,
    })


def regime_report(
    equity: pd.Series,
    regimes: pd.DataFrame,
    *,
    periods_per_year: float = 252.0,
    benchmark: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Performance split by regime.

    The objective is not that a strategy makes money everywhere. It is to know
    *where* it works, where it fails, how badly, and whether the failures are
    the ones its design predicts. A trend follower losing money in a chop is
    behaving correctly; a trend follower losing money in a trend is broken.
    """
    from .metrics import max_drawdown, sharpe

    returns = equity.pct_change().fillna(0.0)
    aligned = regimes.reindex(equity.index)
    bench_returns = (
        benchmark.reindex(equity.index).pct_change().fillna(0.0)
        if benchmark is not None else None
    )

    rows = []
    for label in ("trend", "volatility"):
        for state, mask in aligned.groupby(label).groups.items():
            if state == "warmup":
                continue
            slice_returns = returns.loc[mask]
            if len(slice_returns) < 20:
                continue
            growth = float(np.prod(1.0 + slice_returns.to_numpy()))
            years = len(slice_returns) / periods_per_year
            row = {
                "dimension": label,
                "regime": state,
                "bars": len(slice_returns),
                "share_of_sample": len(slice_returns) / len(returns),
                "total_return": growth - 1.0,
                "annualised": growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else np.nan,
                "sharpe": sharpe(slice_returns, periods_per_year),
                "worst_bar": float(slice_returns.min()),
                "hit_rate": float((slice_returns > 0).mean()),
            }
            if bench_returns is not None:
                bench_growth = float(np.prod(1.0 + bench_returns.loc[mask].to_numpy()))
                row["benchmark_return"] = bench_growth - 1.0
                row["excess"] = (growth - 1.0) - (bench_growth - 1.0)
            rows.append(row)

    crash_mask = aligned["in_crash"].fillna(False)
    if crash_mask.sum() >= 20:
        crash_returns = returns[crash_mask]
        growth = float(np.prod(1.0 + crash_returns.to_numpy()))
        rows.append({
            "dimension": "event", "regime": "benchmark_drawdown_over_20pct",
            "bars": int(crash_mask.sum()),
            "share_of_sample": float(crash_mask.mean()),
            "total_return": growth - 1.0, "annualised": np.nan,
            "sharpe": sharpe(crash_returns, periods_per_year),
            "worst_bar": float(crash_returns.min()),
            "hit_rate": float((crash_returns > 0).mean()),
        })
    return pd.DataFrame(rows)


def era_report(
    equity: pd.Series, *, freq: str = "YE", periods_per_year: float = 252.0
) -> pd.DataFrame:
    """Performance by calendar era — the simplest regime split there is."""
    from .metrics import sharpe

    returns = equity.pct_change().fillna(0.0)
    rows = []
    for period, group in returns.groupby(pd.Grouper(freq=freq)):
        if len(group) < 20:
            continue
        growth = float(np.prod(1.0 + group.to_numpy()))
        rows.append({
            "era": str(period.date())[:4] if freq.startswith("Y") else str(period.date()),
            "bars": len(group),
            "return": growth - 1.0,
            "sharpe": sharpe(group, periods_per_year),
            "worst_bar": float(group.min()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Phase 13 — the battery
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    name: str
    description: str
    stats: Dict[str, float]
    passed: bool
    failures: Tuple[str, ...] = ()


def check_acceptance(stats: Dict[str, float], criteria: Optional[Dict] = None) -> Tuple[bool, Tuple[str, ...]]:
    """Does this result clear the project's floors?"""
    criteria = {**DEFAULT_ACCEPTANCE, **(criteria or {})}
    failures = []
    if stats.get("sharpe", -np.inf) < criteria["min_sharpe"]:
        failures.append(f"sharpe {stats.get('sharpe', float('nan')):.2f} < {criteria['min_sharpe']}")
    if stats.get("max_drawdown", -np.inf) < criteria["max_drawdown"]:
        failures.append(
            f"drawdown {stats.get('max_drawdown', float('nan')):.1%} worse than "
            f"{criteria['max_drawdown']:.0%}"
        )
    if stats.get("n_trades", 0) < criteria["min_trades"]:
        failures.append(f"only {int(stats.get('n_trades', 0))} trades")
    return (not failures), tuple(failures)


def as_utc(date) -> Optional[pd.Timestamp]:
    """Normalise a date to a UTC timestamp, or None.

    Slicing a tz-aware index with a naive string raises, which turns a perfectly
    good robustness scenario into an error row. Call this on any date that came
    in from a config or a command line.
    """
    if date is None:
        return None
    ts = pd.Timestamp(date)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def robustness_battery(
    run: Callable[..., Dict[str, float]],
    *,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    start_dates: Sequence[Optional[str]] = (None,),
    end_dates: Sequence[Optional[str]] = (None,),
    cost_scenarios: Sequence[str] = ("low", "base", "high"),
    extra_slippage_bps: Sequence[float] = (0.0, 10.0, 25.0),
    fill_models: Sequence[str] = ("next_open", "next_close"),
    criteria: Optional[Dict] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run one strategy under every adverse assumption we know how to make.

    ``run`` is called with keyword arguments (``seed``, ``costs``, ``fill_at``,
    ``start``, ``end``) and returns a stats dict. Anything it does not care
    about it can ignore.

    The output is not a leaderboard. It is a survival table: the useful reading
    is *which* scenarios break the strategy, because that tells you what the
    result was really resting on.
    """
    scenarios: List[Scenario] = []

    def add(name: str, description: str, **kwargs):
        try:
            stats = run(**kwargs)
        except Exception as exc:                     # noqa: BLE001 - a failure is a result
            scenarios.append(Scenario(name, description, {}, False, (f"errored: {exc}",)))
            return
        passed, failures = check_acceptance(stats, criteria)
        scenarios.append(Scenario(name, description, stats, passed, failures))
        if verbose:
            mark = "pass" if passed else "FAIL"
            print(
                f"  [{mark}] {name:<28} ${stats.get('final_equity', float('nan')):>9,.0f}  "
                f"sharpe {stats.get('sharpe', float('nan')):>5.2f}  "
                f"maxDD {stats.get('max_drawdown', float('nan')):>6.1%}  "
                f"trades {int(stats.get('n_trades', 0)):>5}"
                + ("" if passed else f"   <- {failures[0]}")
            )

    add("base", "the headline assumptions")
    for name in cost_scenarios:
        if name == "base":
            continue
        add(f"costs:{name}", f"{name} cost scenario", costs=cost_scenario(name))
    for extra in extra_slippage_bps:
        if extra <= 0:
            continue
        worse = replace(COST_SCENARIOS["base"], slippage_bps=COST_SCENARIOS["base"].slippage_bps + extra,
                        name=f"base+{extra:g}bps_slip")
        add(f"slippage:+{extra:g}bps", "base costs with extra slippage", costs=worse)
    for extra in (5.0, 15.0):
        wider = replace(COST_SCENARIOS["base"],
                        half_spread_bps=COST_SCENARIOS["base"].half_spread_bps + extra,
                        name=f"base+{extra:g}bps_spread")
        add(f"spread:+{extra:g}bps", "base costs with a wider spread", costs=wider)
    for seed in seeds:
        add(f"seed:{seed}", "a different random seed", seed=seed)
    for start in start_dates:
        if start is None:
            continue
        add(f"start:{start}", "a later start date", start=as_utc(start))
    for end in end_dates:
        if end is None:
            continue
        add(f"end:{end}", "an earlier end date", end=as_utc(end))
    for fill in fill_models:
        if fill == "next_open":
            continue
        add(f"fill:{fill}", "an alternative execution assumption", fill_at=fill)

    frame = pd.DataFrame([
        {
            "scenario": s.name, "description": s.description, "passed": s.passed,
            "final_equity": s.stats.get("final_equity", np.nan),
            "sharpe": s.stats.get("sharpe", np.nan),
            "max_drawdown": s.stats.get("max_drawdown", np.nan),
            "n_trades": s.stats.get("n_trades", np.nan),
            "total_costs": s.stats.get("total_costs", np.nan),
            "why_failed": "; ".join(s.failures),
        }
        for s in scenarios
    ])
    if verbose and not frame.empty:
        rate = frame["passed"].mean()
        print(f"\n  passed {frame['passed'].sum()}/{len(frame)} scenarios ({rate:.0%})")
        if rate < 0.6:
            print("  A strategy that survives fewer than ~60% of these is fragile, not unlucky.")
    return frame
