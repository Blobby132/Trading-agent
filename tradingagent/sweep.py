"""Sweeping candle interval and trading frequency, honestly.

Two dimensions the rest of the package holds fixed:

* **interval** - the candle width the agent sees (1d, 6h, 1h, ...);
* **frequency** - how often it is allowed to move the book, independent of how
  often it forms a view.

Both are easy to sweep badly. Three traps in particular, each of which this
module is built to avoid rather than to warn about afterwards:

1. *Reusing daily lookbacks on finer bars.* ``mom_lookback=60`` is sixty days
   on daily candles and two and a half days on hourly ones. Every bar count is
   rescaled through :mod:`tradingagent.timescale` before a cell runs, including
   the walk-forward windows themselves.
2. *Comparing intervals over different histories.* Coinbase serves daily bars
   from 2015 and 1-minute bars from much later, so an unconstrained comparison
   would hand the coarse intervals two extra bull markets. Every cell in a
   sweep is clipped to one common date range - :func:`common_range`.
3. *Reporting net return only.* Costs scale with trade count, so a faster
   configuration can look better gross and be worse net, or the reverse. Each
   cell trades its test windows twice - once with the real cost model, once
   frictionless, same selections and same weights - so the gap between the two
   curves is exactly what turnover cost.

Nothing here relaxes the methodology: the walk-forward, the purge, the embargo,
the training-drawdown disqualifier and the one-bar decision lag are the same
objects the daily backtest uses, with their windows expressed in the new
interval's bars.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data import load_prices
from .engine import ExecutionConfig
from .execution import CostModel
from .metrics import deflated_sharpe, summarize
from .optimize import (
    DEFAULT_SEARCH_SPACE,
    WalkForwardConfig,
    WalkForwardResult,
    walk_forward,
)
from .timescale import (
    CRYPTO_SCALES,
    TimeScale,
    coverage_report,
    rescale_search_space,
    rescale_walk_forward,
    throttle,
    trades_per_day,
)


# --------------------------------------------------------------------------- #
# cell definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cell:
    """One point of the interval x frequency grid."""

    interval: str
    rebalance_every: int = 1          # bars the book is held between rebalances
    label: str = ""

    def scale(self, calendar_days: float = 365.0, session_hours: float = 24.0) -> TimeScale:
        return TimeScale(self.interval, calendar_days=calendar_days, session_hours=session_hours)

    @property
    def name(self) -> str:
        return self.label or f"{self.interval}/every{self.rebalance_every}"

    def intended_trades_per_day(self, **kw) -> float:
        return trades_per_day(self.rebalance_every, self.scale(**kw))


def frequency_ladder(interval: str, targets: Sequence[float]) -> List[Cell]:
    """Cells approximating each of ``targets`` rebalances per day at ``interval``.

    Duplicates are dropped: at 6-hourly bars, 8/day and 24/day both collapse to
    "every bar", and running the identical cell twice would inflate the trial
    count the deflated Sharpe has to correct for.
    """
    scale = CRYPTO_SCALES[interval]
    seen: Dict[int, Cell] = {}
    for target in targets:
        every = int(max(1, round(scale.bars_per_day / max(float(target), 1e-9))))
        seen.setdefault(every, Cell(interval, every))
    return [seen[k] for k in sorted(seen)]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def common_range(frames: Dict[str, pd.DataFrame]) -> tuple:
    """The date range every interval in ``frames`` covers.

    Without this the interval comparison is confounded by history length, and
    the coarse intervals win for a reason that has nothing to do with the
    interval.
    """
    start = max(df.index[0] for df in frames.values())
    end = min(df.index[-1] for df in frames.values())
    return start, end


def clip(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def load_cells(
    symbol: str,
    intervals: Sequence[str],
    *,
    start: str = "2015-01-01",
    source: str = "coinbase",
    **kw,
) -> Dict[str, pd.DataFrame]:
    """Load one symbol at several intervals, reporting coverage for each."""
    out: Dict[str, pd.DataFrame] = {}
    for interval in intervals:
        frame = load_prices(symbol, interval=interval, start=start, source=source, **kw)
        out[interval] = frame
    return out


def coverage_table(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Bars, span and completeness per interval - the data-quality audit.

    ``coverage`` below about 0.98 means the series has holes, and holes at fine
    granularity are not neutral: they are concentrated in illiquid periods,
    which are exactly the periods a high-frequency backtest would otherwise
    have lost money in.
    """
    rows = []
    for interval, frame in frames.items():
        row = {"interval": interval}
        row.update(coverage_report(frame, CRYPTO_SCALES[interval]))
        row["start"] = frame.index[0]
        row["end"] = frame.index[-1]
        rows.append(row)
    return pd.DataFrame(rows).set_index("interval")


# --------------------------------------------------------------------------- #
# running a cell
# --------------------------------------------------------------------------- #
@dataclass
class CellResult:
    cell: Cell
    stats: Dict[str, float]
    gross_stats: Dict[str, float]
    result: WalkForwardResult
    scale: TimeScale
    evaluations: int
    seconds: float
    meta: Dict[str, object] = field(default_factory=dict)

    # -- the numbers the sweep report is built from ------------------------ #
    def row(self) -> Dict[str, object]:
        s, g = self.stats, self.gross_stats
        init = float(self.result.exec_config.initial_capital)
        net_final = s.get("final_equity", float("nan"))
        gross_final = g.get("final_equity", float("nan"))
        years = max(float(s.get("bars", 0.0)) / self.scale.bars_per_year, 1e-9)
        costs = float(s.get("total_costs", 0.0)) + float(s.get("financing_paid", 0.0))
        return {
            "interval": self.cell.interval,
            "every": self.cell.rebalance_every,
            "intended_trades_day": self.cell.intended_trades_per_day(),
            "actual_trades_day": float(s.get("n_trades", 0.0)) / (years * 365.0),
            "oos_years": years,
            "final_equity": net_final,
            "gross_final_equity": gross_final,
            "net_return": net_final / init - 1.0,
            "gross_return": gross_final / init - 1.0,
            # Costs paid as a fraction of the STARTING stake. On a compounding
            # account this exceeds 100% routinely and that is not a bug: $150 of
            # fees on a $100 stake that grew to $1,000 is 150% of the stake.
            "cost_drag_pct_capital": costs / max(init, 1e-9),
            "cost_drag_pct_of_gross_profit": (
                costs / max(gross_final - init, 1e-9) if gross_final > init else float("nan")
            ),
            "fees_paid": float(s.get("total_costs", 0.0)),
            "financing_paid": float(s.get("financing_paid", 0.0)),
            "cagr": s.get("cagr"),
            "gross_cagr": g.get("cagr"),
            "sharpe": s.get("sharpe"),
            "gross_sharpe": g.get("sharpe"),
            "max_drawdown": s.get("max_drawdown"),
            "calmar": s.get("calmar"),
            "n_trades": s.get("n_trades"),
            "turnover_per_year": s.get("turnover_per_year"),
            "target_hit": s.get("target_hit"),
            "years_to_target": s.get("years_to_target"),
            "evaluations": self.evaluations,
            "seconds": self.seconds,
        }


def run_cell(
    frame: pd.DataFrame,
    cell: Cell,
    *,
    base_exec: Optional[ExecutionConfig] = None,
    wf_daily: Optional[WalkForwardConfig] = None,
    space_daily: Optional[Dict[str, Sequence]] = None,
    calendar_days: float = 365.0,
    session_hours: float = 24.0,
    verbose: bool = False,
) -> CellResult:
    """Walk-forward one interval/frequency cell, rescaled and cost-audited.

    ``wf_daily`` and ``space_daily`` are given in **daily** bars - the units the
    rest of the package is written in - and rescaled here. Passing an
    already-rescaled config would double-scale it.
    """
    scale = TimeScale(cell.interval, calendar_days=calendar_days, session_hours=session_hours)
    wf_daily = wf_daily or WalkForwardConfig()
    space_daily = space_daily or DEFAULT_SEARCH_SPACE

    wf = replace(rescale_walk_forward(wf_daily, scale), verbose=verbose)
    space = rescale_search_space(space_daily, scale)
    base = replace(
        base_exec or ExecutionConfig(),
        periods_per_year=scale.bars_per_year,
        record_trades=False,   # a per-fill row per candidate per fold dominates runtime
    )

    transform = (lambda w: throttle(w, cell.rebalance_every)) if cell.rebalance_every > 1 else None

    t0 = time.time()
    result = walk_forward(
        frame, base, wf, space, weight_transform=transform, gross_twin=True,
        label=f"sweep:{cell.name}",
    )
    seconds = time.time() - t0

    stats = result.stats()
    gross_equity = result.meta.get("gross_equity")
    gross_stats = _curve_stats(gross_equity, base) if gross_equity is not None else {}
    return CellResult(
        cell=cell, stats=stats, gross_stats=gross_stats, result=result,
        scale=scale, evaluations=result.n_evaluations, seconds=seconds,
    )


def _curve_stats(equity: pd.Series, base: ExecutionConfig) -> Dict[str, float]:
    """Headline statistics for a bare equity curve (the frictionless twin)."""
    from .metrics import cagr, max_drawdown, sharpe

    rets = equity.pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)
    ppy = base.periods_per_year
    return {
        "final_equity": float(equity.iloc[-1]),
        "cagr": cagr(equity, ppy),
        "sharpe": sharpe(rets, ppy),
        "max_drawdown": max_drawdown(equity),
        "bars": float(len(equity)),
    }


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
def run_sweep(
    frames: Dict[str, pd.DataFrame],
    cells: Sequence[Cell],
    *,
    base_exec: Optional[ExecutionConfig] = None,
    wf_daily: Optional[WalkForwardConfig] = None,
    space_daily: Optional[Dict[str, Sequence]] = None,
    align: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every cell and return one row each, with sweep-wide deflation.

    ``align=True`` clips every interval to the range they all cover, so the
    comparison is about the interval and not about who got more history.

    The deflated Sharpe is reported twice per cell, which is the point of doing
    it here rather than per run:

    * ``dsr_cell`` corrects for the configurations that cell alone tried;
    * ``dsr_sweep`` corrects for every configuration tried anywhere in the
      sweep. Picking the best cell out of a grid *is* a search, and the second
      number is the one that applies to the winner.
    """
    if align:
        start, end = common_range(frames)
        frames = {k: clip(v, start, end) for k, v in frames.items()}
        if verbose:
            print(f"[sweep] common range {start} -> {end}")

    rows: List[Dict[str, object]] = []
    results: List[CellResult] = []
    for cell in cells:
        if verbose:
            print(f"[sweep] {cell.name} ...", flush=True)
        res = run_cell(
            frames[cell.interval], cell, base_exec=base_exec,
            wf_daily=wf_daily, space_daily=space_daily, verbose=False,
        )
        results.append(res)
        rows.append(res.row())
        if verbose:
            r = rows[-1]
            print(
                f"[sweep] {cell.name}: net ${r['final_equity']:,.0f} "
                f"gross ${r['gross_final_equity']:,.0f} "
                f"Sharpe {r['sharpe']:.2f}  maxDD {r['max_drawdown']:.1%}  "
                f"costs {r['cost_drag_pct_capital']:.0%} of stake  "
                f"({r['seconds']:.0f}s)", flush=True
            )

    table = pd.DataFrame(rows)
    total_trials = int(table["evaluations"].sum())
    table["dsr_cell"] = [
        deflated_sharpe(r.stats.get("sharpe", float("nan")), r.evaluations,
                        int(r.stats.get("bars", 0)), r.scale.bars_per_year)
        for r in results
    ]
    table["dsr_sweep"] = [
        deflated_sharpe(r.stats.get("sharpe", float("nan")), total_trials,
                        int(r.stats.get("bars", 0)), r.scale.bars_per_year)
        for r in results
    ]
    table.attrs["total_trials"] = total_trials
    table.attrs["results"] = results
    return table


# --------------------------------------------------------------------------- #
# the arithmetic that decides most of this in advance
# --------------------------------------------------------------------------- #
def breakeven_edge(
    cost_model: CostModel, trades_per_day_: float, *, calendar_days: float = 365.0
) -> Dict[str, float]:
    """Gross edge per trade needed to break even at a given trade rate.

    Worth computing before any backtest, because it is arithmetic rather than
    evidence and it rules out most of the high-frequency grid on its own. A
    round trip pays the per-side cost twice; at N round trips a day the annual
    cost is ``2 * cost * N * days``, and the strategy has to out-earn that
    before it has earned anything at all.

    ``annual_cost_if_full_turnover`` is that annual figure as a fraction of
    equity, on the assumption that each rebalance replaces the whole book. Real
    rebalances move less than that - the dust filter suppresses small ones - so
    treat it as the ceiling, not the estimate. Even as a ceiling it is the
    number that decides the fast end of the grid: at four rebalances a day it
    says the strategy must earn 438% a year gross before it has earned a cent.
    """
    per_side = (
        cost_model.fee_bps + cost_model.half_spread_bps + cost_model.slippage_bps
    ) / 1e4
    round_trip = 2.0 * per_side
    annual = round_trip * trades_per_day_ * calendar_days
    return {
        "per_side_bps": per_side * 1e4,
        "round_trip_bps": round_trip * 1e4,
        "round_trips_per_year": trades_per_day_ * calendar_days,
        "annual_cost_if_full_turnover": annual,
        "required_gross_edge_per_trade_bps": round_trip * 1e4,
    }


def breakeven_table(
    cost_model: CostModel, rates: Sequence[float], *, calendar_days: float = 365.0
) -> pd.DataFrame:
    rows = [
        {"trades_per_day": rate, **breakeven_edge(cost_model, rate, calendar_days=calendar_days)}
        for rate in rates
    ]
    return pd.DataFrame(rows).set_index("trades_per_day")
