"""Quantifying survivorship bias instead of waving at it.

A universe of names that are liquid *today* omits every company that was large
and then failed. That inflates any backtest run over it. The honest responses,
in order of preference:

1. **Put the dead names back.** :data:`tradingagent.universe.DELISTED_LARGE_CAP`
   lists companies that failed, were acquired, or went private after being
   large, and :func:`compare_universes` measures what including them costs.
   Yahoo serves their history; Nasdaq's quote API does not, because it is a
   live-quote service.
2. **When you cannot get the dead names, bound the damage.**
   :func:`stress_test` injects synthetic failures into a survivor-only universe -
   names that go to zero over a few weeks, concentrated where real failures
   concentrate, at the bottom of the momentum ranking - and reports how far the
   result moves. That converts "there is some bias" into a number with a stated
   assumption.

Neither is a substitute for point-in-time index membership. Both beat a
disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .engine import BacktestEngine, ExecutionConfig
from .metrics import summarize
from .risk import RiskConfig
from .universe import Panel


@dataclass
class StressResult:
    """What synthetic failures did to a strategy and to its benchmark."""

    failure_rate: float               # annual probability a name dies
    n_failures: int
    strategy_equity: float
    benchmark_equity: float
    strategy_drag: float              # fraction of final equity lost
    benchmark_drag: float
    relative_drag: float              # strategy drag minus benchmark drag

    def __str__(self) -> str:
        return (
            f"{self.failure_rate:>5.1%}/yr  {self.n_failures:>3d} failures  "
            f"strategy ${self.strategy_equity:>8,.0f} ({self.strategy_drag:+6.1%})  "
            f"benchmark ${self.benchmark_equity:>8,.0f} ({self.benchmark_drag:+6.1%})  "
            f"relative {self.relative_drag:+6.1%}"
        )


def inject_failures(
    panel: Panel,
    *,
    failure_rate: float = 0.02,
    rank_signal: Optional[pd.DataFrame] = None,
    concentration: float = 4.0,
    decline_bars: int = 21,
    seed: int = 0,
    periods_per_year: float = 252.0,
) -> tuple[Panel, List[tuple[str, pd.Timestamp]]]:
    """Kill off names at random, weighted toward the bottom of ``rank_signal``.

    A failing company does not gap to zero out of nowhere: it declines hard over
    weeks and then stops trading. Each victim is faded to near-zero over
    ``decline_bars`` and set to NaN afterwards, which is exactly the delisting
    the engine already knows how to handle.

    ``concentration`` tilts which names die. Real failures are not uniform over
    the cross-section - they land overwhelmingly on names that have already been
    falling - so with a momentum signal supplied, a value of 4 makes the weakest
    decile roughly four times as likely to be chosen as the average name.
    Setting it to 0 spreads failures uniformly, which is the pessimistic case
    for a momentum strategy because it lets winners die too.
    """
    rng = np.random.default_rng(seed)
    close = panel.close.copy()
    open_ = panel.open.copy()
    high = panel.high.copy()
    low = panel.low.copy()
    volume = panel.volume.copy()

    years = len(panel) / periods_per_year
    expected = failure_rate * years * len(panel.symbols)
    n_failures = int(rng.poisson(max(expected, 0.0)))
    if n_failures == 0:
        return panel, []

    # weight the draw toward names that are already weak, if a signal is given
    weights = np.ones(len(panel.symbols))
    if rank_signal is not None and concentration > 0:
        avg = rank_signal.reindex(columns=panel.symbols).mean(axis=0)
        ranks = avg.rank(pct=True).fillna(0.5).to_numpy()   # 0 = weakest
        weights = np.exp(-concentration * ranks)
    weights = weights / weights.sum()

    victims = rng.choice(
        len(panel.symbols), size=min(n_failures, len(panel.symbols)), replace=False, p=weights
    )
    earliest = max(int(0.1 * len(panel)), decline_bars + 1)
    events: List[tuple[str, pd.Timestamp]] = []

    for j in victims:
        sym = panel.symbols[j]
        valid = close[sym].dropna()
        if len(valid) < earliest + decline_bars:
            continue
        start_pos = int(rng.integers(earliest, len(panel) - 1))
        stop_pos = min(start_pos + decline_bars, len(panel) - 1)
        idx = panel.index

        # fade to 1% of the prevailing price, then stop trading altogether
        span = stop_pos - start_pos
        if span <= 0:
            continue
        base = close[sym].iloc[start_pos]
        if not np.isfinite(base):
            continue
        path = base * np.geomspace(1.0, 0.01, span)
        for frame in (close, open_, high, low):
            frame.iloc[start_pos:stop_pos, frame.columns.get_loc(sym)] = path
        for frame in (close, open_, high, low, volume):
            frame.iloc[stop_pos:, frame.columns.get_loc(sym)] = np.nan
        events.append((sym, idx[start_pos]))

    return Panel(open_, high, low, close, volume), events


def stress_test(
    panel: Panel,
    weight_fn: Callable[[Panel], pd.DataFrame],
    *,
    failure_rates: Sequence[float] = (0.0, 0.01, 0.02, 0.04),
    n_trials: int = 5,
    exec_config: ExecutionConfig | None = None,
    risk_config: RiskConfig | None = None,
    rank_signal: Optional[pd.DataFrame] = None,
    concentration: float = 4.0,
    start: Optional[pd.Timestamp] = None,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-run a strategy over universes with synthetic failures injected.

    ``weight_fn`` maps a panel to portfolio weights, so the strategy is rebuilt
    on the damaged universe rather than having its old weights replayed - which
    is the point: a name that has started failing should fall out of the
    ranking on its own.

    The benchmark is an equal-weight basket of the same damaged universe, so the
    ``relative`` column isolates what failures cost the *strategy* beyond what
    they cost the market it is being compared against.
    """
    exec_config = exec_config or ExecutionConfig(initial_capital=100.0, periods_per_year=252.0)
    risk_config = risk_config or RiskConfig(
        target_vol=0.0, atr_stop_mult=0.0, max_drawdown_stop=0.0, reentry_lockout_bars=0
    )

    def run(p: Panel, weights: pd.DataFrame) -> float:
        frames = p.to_frames()
        if start is not None:
            frames = {k: v.loc[start:] for k, v in frames.items()}
            weights = weights.loc[start:]
        return float(summarize(BacktestEngine(exec_config, risk_config).run(frames, weights))["final_equity"])

    def equal_weight(p: Panel) -> pd.DataFrame:
        live = p.tradeable().astype(float)
        w = live.div(live.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        keep = np.zeros(len(w), dtype=bool)
        keep[::21] = True
        return w.where(pd.Series(keep, index=w.index), np.nan).ffill().fillna(0.0)

    clean_strategy = run(panel, weight_fn(panel))
    clean_benchmark = run(panel, equal_weight(panel))

    rows: List[dict] = []
    for rate in failure_rates:
        trials = 1 if rate == 0 else n_trials
        for trial in range(trials):
            damaged, events = inject_failures(
                panel,
                failure_rate=rate,
                rank_signal=rank_signal,
                concentration=concentration,
                seed=seed + trial * 101,
                periods_per_year=exec_config.periods_per_year,
            )
            strat = run(damaged, weight_fn(damaged))
            bench = run(damaged, equal_weight(damaged))
            result = StressResult(
                failure_rate=rate,
                n_failures=len(events),
                strategy_equity=strat,
                benchmark_equity=bench,
                strategy_drag=strat / clean_strategy - 1.0,
                benchmark_drag=bench / clean_benchmark - 1.0,
                relative_drag=(strat / clean_strategy) - (bench / clean_benchmark),
            )
            rows.append(vars(result))
            if verbose and trial == 0:
                print(f"  {result}")

    frame = pd.DataFrame(rows)
    return (
        frame.groupby("failure_rate")
        .agg(
            trials=("n_failures", "size"),
            failures=("n_failures", "mean"),
            strategy=("strategy_equity", "median"),
            benchmark=("benchmark_equity", "median"),
            strategy_drag=("strategy_drag", "median"),
            benchmark_drag=("benchmark_drag", "median"),
            relative_drag=("relative_drag", "median"),
        )
        .reset_index()
    )


def compare_universes(
    survivors: Panel,
    with_delisted: Panel,
    weight_fn: Callable[[Panel], pd.DataFrame],
    *,
    exec_config: ExecutionConfig | None = None,
    risk_config: RiskConfig | None = None,
    start: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    """The direct measurement, when the dead names are actually available.

    Runs the identical strategy over a survivor-only universe and over the same
    universe with failed and acquired names restored. The gap is survivorship
    bias, measured rather than assumed.
    """
    exec_config = exec_config or ExecutionConfig(initial_capital=100.0, periods_per_year=252.0)
    risk_config = risk_config or RiskConfig(
        target_vol=0.0, atr_stop_mult=0.0, max_drawdown_stop=0.0, reentry_lockout_bars=0
    )

    def run(p: Panel) -> Dict[str, float]:
        weights = weight_fn(p)
        frames = p.to_frames()
        if start is not None:
            frames = {k: v.loc[start:] for k, v in frames.items()}
            weights = weights.loc[start:]
        return summarize(BacktestEngine(exec_config, risk_config).run(frames, weights))

    a, b = run(survivors), run(with_delisted)
    return {
        "survivors_only_equity": a["final_equity"],
        "with_delisted_equity": b["final_equity"],
        "survivorship_bias": a["final_equity"] / max(b["final_equity"], 1e-9) - 1.0,
        "survivors_only_sharpe": a["sharpe"],
        "with_delisted_sharpe": b["sharpe"],
        "n_survivors": float(len(survivors.symbols)),
        "n_with_delisted": float(len(with_delisted.symbols)),
    }
