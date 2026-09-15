"""Walk-forward parameter selection.

Choosing the best configuration over the whole history and then quoting its
return is how backtests lie. Everything here is built to avoid that:

* parameters are chosen on a **training window** and then traded, untouched,
  on the **test window that follows it**;
* the two windows are separated by an *embargo* so rolling indicators cannot
  smear information across the boundary;
* equity carries from one test window to the next, so the stitched curve is a
  single, continuously-compounding account - which is exactly the thing the
  $100 -> $1,000 question is about;
* the reported statistics come only from the stitched out-of-sample curve.

``combine_top_k`` goes one step further: instead of trading the single best
configuration from the training window, it trades the average of the best *k*,
which is far less sensitive to a lucky parameter cell.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .agent import AgentConfig, PortfolioAgent, TradingAgent
from .engine import BacktestEngine, BacktestResult, ExecutionConfig
from .holdout import Holdout
from .ledger import ResearchLedger
from .metrics import summarize
from .risk import RiskConfig
from .strategies import DEFAULT_ENSEMBLE

# --------------------------------------------------------------------------- #
# search space
# --------------------------------------------------------------------------- #
STRATEGY_SUBSETS: List[List[str]] = [
    list(DEFAULT_ENSEMBLE),
    ["ema_trend", "donchian", "ts_momentum", "bollinger_breakout"],   # trend only
    ["ema_trend", "donchian", "ts_momentum"],
    ["donchian", "bollinger_breakout"],                               # breakout only
    ["ema_trend", "ts_momentum", "rsi_pullback"],
    ["ema_trend", "donchian", "ts_momentum", "mean_reversion"],
]

#: The **legacy** space: 20 parameters, 3.3 billion combinations. Kept so old
#: results can be reproduced, and kept out of the default because a space that
#: large searched over ~3,300 bars is a machine for finding coincidences. Every
#: extra knob multiplies the number of ways to fit this particular sample.
LEGACY_WIDE_SEARCH_SPACE: Dict[str, Sequence] = {
    # ensemble
    "strategies": list(range(len(STRATEGY_SUBSETS))),
    "weighting": ["adaptive", "equal", "best"],
    "perf_lookback": [60, 120, 250],
    "softmax_temp": [2.0, 4.0, 8.0],
    "regime_filter": [0, 1],
    "regime_trend": [100, 200],
    "signal_smooth": [1, 3, 5, 10],
    "allow_short": [0, 1],
    # per-strategy shape
    "ema_fast": [10, 20, 50],
    "ema_slow": [100, 150, 200],
    "donchian_entry": [20, 40, 55],
    "mom_lookback": [40, 60, 120],
    # risk
    "target_vol": [0.30, 0.50, 0.80],
    "max_leverage": [1.0, 1.5, 2.0, 3.0],
    "atr_stop_mult": [0.0, 4.0, 6.0, 8.0],
    "trail_stop": [0, 1],
    "max_drawdown_stop": [0.25, 0.35, 0.50],
    # execution
    "min_trade_frac": [0.05, 0.10, 0.20],
    # multi-asset only (ignored when a single symbol is backtested)
    "allocation": ["inverse_vol", "equal", "momentum"],
    "max_positions": [0, 2, 3],
}

#: The default. Every entry is a choice with an economic argument behind it -
#: *what* to trade, *which way*, *how much*, and *when to stop* - and the
#: nuisance knobs that only exist because some function needed a number are
#: pinned at their defaults rather than searched.
#:
#: Roughly 1,500 combinations instead of 3.3 billion. That is not a cosmetic
#: reduction: the number of distinct ways to fit a sample is what a
#: multiple-testing correction is correcting for, and the deflated Sharpe
#: improves because the search is smaller, not because the strategy got better.
DEFAULT_SEARCH_SPACE: Dict[str, Sequence] = {
    # what to trade, and how to combine the views
    "strategies": list(range(len(STRATEGY_SUBSETS))),
    "weighting": ["adaptive", "equal"],   # "best" is winner-take-all: dropped
    "perf_lookback": [60, 250],
    "regime_filter": [0, 1],
    "allow_short": [0, 1],
    "signal_smooth": [1, 5],
    # how much to hold
    "target_vol": [0.30, 0.50, 0.80],
    "max_leverage": [1.0, 2.0],
    # when to stop holding it
    "atr_stop_mult": [0.0, 6.0],
    # -- pinned: nuisance parameters, not economic choices ------------------
    "softmax_temp": [4.0],
    "regime_trend": [200],
    "ema_fast": [20],
    "ema_slow": [100],
    "donchian_entry": [40],
    "mom_lookback": [60],
    "trail_stop": [1],
    "max_drawdown_stop": [0.35],
    "min_trade_frac": [0.10],
    "allocation": ["inverse_vol"],
    "max_positions": [0],
}


def space_size(space: Dict[str, Sequence]) -> int:
    """How many distinct configurations a space contains."""
    return int(np.prod([len(v) for v in space.values()]))


def params_to_configs(
    params: Dict, base_exec: ExecutionConfig
) -> Tuple[AgentConfig, RiskConfig, ExecutionConfig]:
    """Turn one sampled point of the search space into the three config objects."""
    names = STRATEGY_SUBSETS[int(params["strategies"])]
    strategy_params: Dict[str, dict] = {}
    if "ema_trend" in names:
        strategy_params["ema_trend"] = {"fast": int(params["ema_fast"]), "slow": int(params["ema_slow"])}
    if "donchian" in names:
        entry = int(params["donchian_entry"])
        strategy_params["donchian"] = {"entry": entry, "exit": max(5, entry // 3)}
    if "ts_momentum" in names:
        strategy_params["ts_momentum"] = {"lookback": int(params["mom_lookback"])}

    agent_cfg = AgentConfig(
        strategies=names,
        strategy_params=strategy_params,
        weighting=str(params["weighting"]),
        perf_lookback=int(params["perf_lookback"]),
        softmax_temp=float(params["softmax_temp"]),
        regime_filter=bool(int(params["regime_filter"])),
        regime_trend=int(params["regime_trend"]),
        allow_short=bool(int(params["allow_short"])),
        signal_smooth=int(params["signal_smooth"]),
        signal_deadband=float(params.get("signal_deadband", 0.0)),
        signal_shape=float(params.get("signal_shape", 1.0)),
        trend_floor=float(params.get("trend_floor", 0.0)),
        trend_floor_lookback=int(params.get("trend_floor_lookback", 252)),
        periods_per_year=base_exec.periods_per_year,
    )
    risk_cfg = RiskConfig(
        target_vol=float(params["target_vol"]),
        max_leverage=float(params["max_leverage"]),
        atr_stop_mult=float(params["atr_stop_mult"]),
        trail_stop=bool(int(params["trail_stop"])),
        max_drawdown_stop=float(params["max_drawdown_stop"]),
    )
    exec_cfg = replace(
        base_exec,
        min_trade_frac=float(params["min_trade_frac"]),
        max_leverage=float(params["max_leverage"]),
    )
    return agent_cfg, risk_cfg, exec_cfg


# --------------------------------------------------------------------------- #
# objectives
# --------------------------------------------------------------------------- #
def _finite(x: float, default: float = -1e9) -> float:
    return float(x) if x is not None and np.isfinite(x) else default


def obj_sharpe(stats: Dict[str, float]) -> float:
    return _finite(stats.get("sharpe"))


def obj_calmar(stats: Dict[str, float]) -> float:
    """Growth per unit of pain, with the denominator floored.

    A raw CAGR/maxDD ratio explodes when a window happens to have a tiny
    drawdown, and the optimiser then chases that artefact; flooring the
    drawdown at 15% keeps the ranking sane.
    """
    g = _finite(stats.get("cagr"), -10.0)
    mdd = abs(_finite(stats.get("max_drawdown"), -1.0))
    return g / max(mdd, 0.15)


def obj_growth(stats: Dict[str, float]) -> float:
    """Log terminal wealth - pure compounding, drawdown-blind."""
    final = stats.get("final_equity", 0.0)
    init = max(stats.get("initial_equity", 1.0), 1e-9)
    if final <= 0:
        return -1e9
    return float(np.log(final / init))


def obj_target_growth(stats: Dict[str, float]) -> float:
    """Compounding, but with ruin and deep drawdowns priced in.

    This is the default because it matches the brief: get to the goal fast,
    without a path that would have stopped you out - psychologically or
    literally - before you got there.
    """
    if stats.get("bust"):
        return -1e9
    g = obj_growth(stats)
    mdd = abs(_finite(stats.get("max_drawdown"), -1.0))
    excess_dd = max(0.0, mdd - 0.25)
    trades = stats.get("n_trades", 0)
    thin = 0.5 if trades < 10 else 0.0   # a handful of trades is not evidence
    return g - 8.0 * excess_dd**2 - thin


OBJECTIVES: Dict[str, Callable[[Dict[str, float]], float]] = {
    "sharpe": obj_sharpe,
    "calmar": obj_calmar,
    "growth": obj_growth,
    "target_growth": obj_target_growth,
}


# --------------------------------------------------------------------------- #
# evaluation of a single configuration
# --------------------------------------------------------------------------- #
def sized_weight_for(
    data: pd.DataFrame | Dict[str, pd.DataFrame], params: Dict, base_exec: ExecutionConfig
) -> Tuple[pd.Series | pd.DataFrame, ExecutionConfig, RiskConfig]:
    """Full-history sized weight for one configuration (causal at every bar)."""
    agent_cfg, risk_cfg, exec_cfg = params_to_configs(params, base_exec)
    if isinstance(data, dict):
        agent = PortfolioAgent(
            list(data),
            agent_cfg,
            risk_cfg,
            allocation=str(params.get("allocation", "inverse_vol")),
            max_positions=int(params.get("max_positions", 0)),
        )
        return agent.target_weights(data), exec_cfg, risk_cfg
    return TradingAgent(agent_cfg, risk_cfg).sized_weight(data), exec_cfg, risk_cfg


def slice_data(data: pd.DataFrame | Dict[str, pd.DataFrame], window: slice):
    if isinstance(data, dict):
        return {k: v.iloc[window] for k, v in data.items()}
    return data.iloc[window]


def run_window(
    data: pd.DataFrame | Dict[str, pd.DataFrame],
    weights: pd.Series | pd.DataFrame,
    window: slice,
    exec_cfg: ExecutionConfig,
    risk_cfg: RiskConfig,
    initial_capital: float,
) -> BacktestResult:
    """Trade a pre-computed weight series over one slice of the history."""
    cfg = replace(exec_cfg, initial_capital=float(initial_capital))
    engine = BacktestEngine(cfg, risk_cfg)
    return engine.run(slice_data(data, window), weights.iloc[window])


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def sample_params(rng: np.random.Generator, space: Dict[str, Sequence]) -> Dict:
    return {k: v[int(rng.integers(len(v)))] for k, v in space.items()}


def sample_unique(
    n: int, space: Dict[str, Sequence], seed: int = 0, max_tries_factor: int = 20
) -> List[Dict]:
    """``n`` distinct configurations, or every configuration if the grid is small."""
    total = int(np.prod([len(v) for v in space.values()]))
    if total <= n:
        keys = list(space)
        return [dict(zip(keys, combo)) for combo in itertools.product(*[space[k] for k in keys])]
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    for _ in range(n * max_tries_factor):
        p = sample_params(rng, space)
        key = tuple(sorted(p.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardConfig:
    train_bars: int = 730          # ~2 years of daily bars to fit on
    test_bars: int = 182           # ~6 months traded out of sample
    embargo_bars: int = 10         # gap so rolling windows cannot leak across
    n_candidates: int = 150        # random configurations tried per fold
    top_k: int = 5                 # configurations blended into the traded signal
    objective: str = "target_growth"
    max_train_drawdown: float = 0.45   # a candidate that drew down more than this
                                       # in training is disqualified, however good
                                       # its return was
    seed: int = 0
    verbose: bool = True


@dataclass
class WalkForwardResult:
    equity: pd.Series
    returns: pd.Series
    folds: pd.DataFrame
    chosen: List[List[Dict]]
    exec_config: ExecutionConfig
    risk_config: RiskConfig
    weights: pd.DataFrame
    n_evaluations: int = 0
    meta: Dict[str, object] = field(default_factory=dict)

    def stats(self, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
        result = BacktestResult(
            equity=self.equity,
            returns=self.returns,
            weights=self.weights,
            target_weights=self.weights,
            trades=self.meta.get("trades", pd.DataFrame()),
            costs=self.meta.get("costs", pd.DataFrame(index=self.equity.index)),
            exec_config=self.exec_config,
            risk_config=self.risk_config,
            # A search runs with record_trades=False because building a row per
            # fill dominates the runtime, but the engine still counts fills and
            # notional. Carrying those counters forward is what keeps trade
            # counts and turnover reportable out of a walk-forward - without
            # them a cost-vs-frequency comparison has no denominator.
            meta={
                "bust": bool(self.equity.iloc[-1] <= self.exec_config.ruin_equity),
                "n_trades": int(self.meta.get("n_trades", 0)),
                "traded_notional": float(self.meta.get("traded_notional", 0.0)),
                "kill_switch_events": int(self.meta.get("kill_switch_events", 0)),
            },
        )
        return summarize(result, benchmark=benchmark)


def walk_forward(
    data: pd.DataFrame | Dict[str, pd.DataFrame],
    base_exec: ExecutionConfig | None = None,
    wf: WalkForwardConfig | None = None,
    space: Dict[str, Sequence] | None = None,
    *,
    holdout: Optional["Holdout"] = None,
    ledger: Optional["ResearchLedger"] = None,
    label: str = "walk_forward",
    weight_transform: Optional[Callable] = None,
    gross_twin: bool = False,
    selection: str = "best",
) -> WalkForwardResult:
    """Fit, step forward, trade, repeat - and report only the traded part.

    Pass ``holdout`` to make the reserved era unreachable: the search raises
    rather than quietly optimising over data that is supposed to be unseen.
    Pass ``ledger`` to record the run, including how many candidates were
    evaluated, so repeated searching stays visible.

    Pass ``weight_transform`` to post-process every candidate's sized weight
    before it is scored or traded - this is how a trading-frequency throttle is
    applied to a whole sweep cell without adding an axis to the search space.
    It must be causal; :func:`tradingagent.timescale.throttle` is.

    ``selection`` is normally ``"best"`` - take the top-k by training score.
    Pass ``"random"`` to pick k candidates at random on every fold instead. That
    is not a strategy, it is a **null**: it leaves the data and the candidate
    pool untouched and removes only the optimiser's judgement, so the gap
    between the two says what the selection step is actually worth. See
    :mod:`tradingagent.falsify`.

    Pass ``gross_twin=True`` to trade each test window a second time with costs
    switched off, chaining its own capital. The selection, the weights and the
    windows are identical by construction, so the difference between the two
    curves is exactly what fees, slippage and financing took - which is the only
    honest way to say whether a faster configuration earned its turnover. The
    gross curve lands in ``meta["gross_equity"]``.

    ``data`` is either one OHLCV frame or a dict of them (a portfolio sharing
    one account, which must already be index-aligned - see
    :func:`tradingagent.data.align_universe`).
    """
    base_exec = base_exec or ExecutionConfig()
    wf = wf or WalkForwardConfig()
    space = space or DEFAULT_SEARCH_SPACE
    if holdout is not None:
        holdout.guard(data if not isinstance(data, dict) else next(iter(data.values())),
                      what="walk_forward")
    objective = OBJECTIVES[wf.objective]

    candidates = sample_unique(wf.n_candidates, space, seed=wf.seed)
    index = (data[next(iter(data))] if isinstance(data, dict) else data).index
    n = len(index)

    # every candidate's sized weight is computed once over the whole history;
    # each value still only depends on bars at or before its own timestamp, so
    # slicing it per fold is safe and saves recomputing the panel every fold
    cached = [sized_weight_for(data, p, base_exec) for p in candidates]
    if weight_transform is not None:
        cached = [(weight_transform(w), ec, rc) for w, ec, rc in cached]

    equity_pieces: List[pd.Series] = []
    weight_pieces: List[pd.Series | pd.DataFrame] = []
    trade_pieces: List[pd.DataFrame] = []
    cost_pieces: List[pd.DataFrame] = []
    fold_rows: List[dict] = []
    chosen_per_fold: List[List[Dict]] = []

    capital = float(base_exec.initial_capital)
    # The gross twin runs the same decisions through a frictionless account.
    # Every cost channel has to be zeroed, not just fees: a leveraged book pays
    # financing whether or not it trades, and leaving that in would understate
    # the cost of holding while overstating the cost of turnover.
    free_exec = replace(
        base_exec,
        costs=replace(
            base_exec.cost_model(),
            fee_bps=0.0, half_spread_bps=0.0, slippage_bps=0.0,
            impact_bps_at_full=0.0, borrow_rate=0.0, short_rate=0.0,
            name="frictionless",
        ),
    )
    gross_pieces: List[pd.Series] = []
    gross_capital = float(base_exec.initial_capital)
    traded_notional = 0.0
    n_trades = 0
    kill_switch_events = 0
    start = 0
    fold_id = 0
    evaluations = 0
    last_exec, last_risk = base_exec, RiskConfig()

    while start + wf.train_bars + wf.embargo_bars + 1 < n:
        train = slice(start, start + wf.train_bars)
        test_start = start + wf.train_bars + wf.embargo_bars
        test_end = min(test_start + wf.test_bars, n)
        if test_end - test_start < 5:
            break
        test = slice(test_start, test_end)

        # ---- fit: score every candidate on the training window ---------- #
        scored: List[Tuple[float, int]] = []
        for idx, (w, ec, rc) in enumerate(cached):
            res = run_window(data, w, train, ec, rc, base_exec.initial_capital)
            stats = summarize(res)
            score = objective(stats)
            # A soft disqualification rather than a filter: if every candidate
            # breaches the drawdown limit the ranking still works, it just
            # ranks a field of bad options.
            if abs(_finite(stats.get("max_drawdown"), -1.0)) > wf.max_train_drawdown:
                score -= 1000.0
            scored.append((score, idx))
            evaluations += 1
        scored.sort(key=lambda t: t[0], reverse=True)
        if selection == "random":
            # Deliberately ignore the ranking. The fold is still scored, so the
            # candidate pool and the training cost are identical - only the
            # choice is thrown away.
            picker = np.random.default_rng(wf.seed * 100003 + fold_id)
            top = [int(i) for i in picker.choice(len(scored), size=max(1, wf.top_k), replace=False)]
        elif selection == "best":
            top = [idx for _, idx in scored[: max(1, wf.top_k)]]
        else:
            raise ValueError(f"selection must be 'best' or 'random', got {selection!r}")
        chosen_per_fold.append([candidates[i] for i in top])

        # ---- trade: average the top-k weights over the test window ------ #
        blended = sum(cached[i][0] for i in top) / len(top)
        ec = cached[top[0]][1]
        rc = cached[top[0]][2]
        # blending configurations means blending their leverage caps too
        ec = replace(ec, max_leverage=float(np.mean([cached[i][1].max_leverage for i in top])))
        rc = replace(rc, max_leverage=ec.max_leverage)
        last_exec, last_risk = ec, rc

        res = run_window(data, blended, test, ec, rc, capital)
        if gross_twin:
            gec = replace(free_exec, min_trade_frac=ec.min_trade_frac, max_leverage=ec.max_leverage)
            gres = run_window(data, blended, test, gec, rc, gross_capital)
            gross_pieces.append(gres.equity)
            gross_capital = max(float(gres.equity.iloc[-1]), 0.0)
        n_trades += int(res.meta.get("n_trades", 0))
        traded_notional += float(res.meta.get("traded_notional", 0.0))
        kill_switch_events += int(res.meta.get("kill_switch_events", 0))
        equity_pieces.append(res.equity)
        weight_pieces.append(blended.iloc[test])
        cost_pieces.append(res.costs)
        if not res.trades.empty:
            trade_pieces.append(res.trades)

        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": index[train.start],
                "train_end": index[train.stop - 1],
                "test_start": index[test_start],
                "test_end": index[test_end - 1],
                "start_equity": capital,
                "end_equity": float(res.equity.iloc[-1]),
                "return": float(res.equity.iloc[-1] / capital - 1.0),
                "max_drawdown": float((res.equity / res.equity.cummax() - 1.0).min()),
                "trades": int(len(res.trades)),
                "train_score": float(scored[0][0]),
                "best_params": candidates[top[0]],
            }
        )
        if wf.verbose:
            row = fold_rows[-1]
            print(
                f"[fold {fold_id:>2}] test {row['test_start'].date()} -> {row['test_end'].date()}  "
                f"${row['start_equity']:>9,.2f} -> ${row['end_equity']:>9,.2f}  "
                f"({row['return']:+7.1%}, maxDD {row['max_drawdown']:6.1%})"
            )

        capital = max(float(res.equity.iloc[-1]), 0.0)
        if capital <= base_exec.ruin_equity:
            if wf.verbose:
                print(f"[fold {fold_id}] account ruined - stopping walk-forward")
            break
        start += wf.test_bars
        fold_id += 1

    if not equity_pieces:
        raise ValueError(
            f"not enough data for a walk-forward: {n} bars, needs > "
            f"{wf.train_bars + wf.embargo_bars + wf.test_bars}"
        )

    equity = pd.concat(equity_pieces)
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    weights = pd.concat(weight_pieces)
    weights = weights[~weights.index.duplicated(keep="last")].sort_index()
    if isinstance(weights, pd.Series):
        weights = weights.to_frame("asset")

    result = WalkForwardResult(
        equity=equity.rename("equity"),
        returns=equity.pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0),
        folds=pd.DataFrame(fold_rows),
        chosen=chosen_per_fold,
        exec_config=last_exec,
        risk_config=last_risk,
        weights=weights,
        n_evaluations=evaluations,
        meta={
            "trades": pd.concat(trade_pieces) if trade_pieces else pd.DataFrame(),
            "costs": pd.concat(cost_pieces) if cost_pieces else pd.DataFrame(index=equity.index),
            "n_candidates": len(candidates),
            "n_trades": n_trades,
            "traded_notional": traded_notional,
            "kill_switch_events": kill_switch_events,
            "gross_equity": (
                pd.concat(gross_pieces)[lambda e: ~e.index.duplicated(keep="last")].sort_index()
                if gross_pieces
                else None
            ),
        },
    )
    if ledger is not None:
        # a walk-forward *selects* a configuration on every fold, so it counts
        # as selection: its candidate count belongs in the multiple-testing total
        ledger.record(
            label, result.stats(),
            universe=",".join(data) if isinstance(data, dict) else "single_asset",
            n_symbols=len(data) if isinstance(data, dict) else 1,
            dataset_start=index[0], dataset_end=index[-1],
            candidate_count=evaluations, parameters=dict(chosen_per_fold[-1][0]) if chosen_per_fold else {},
            seed=wf.seed, holdout_status="holdout" if holdout is None else "development",
            objective=wf.objective, cost_scenario=base_exec.cost_model().name,
            used_for_selection=True,
            notes=f"{len(fold_rows)} folds, top_k={wf.top_k}",
        )
    return result


# --------------------------------------------------------------------------- #
# searching until the goal is met
# --------------------------------------------------------------------------- #
@dataclass
class TargetSearchResult:
    """Outcome of :func:`search_until_target`."""

    reached: bool
    attempts: pd.DataFrame
    best: Optional[WalkForwardResult]
    best_label: str
    target: float

    def summary(self) -> str:
        lines = [
            f"== search for ${self.target:,.0f} ==",
            f"  attempts run      {len(self.attempts)}",
        ]
        if self.reached:
            row = self.attempts[self.attempts["target_hit"]].iloc[0]
            lines += [
                f"  TARGET REACHED    {row['label']} (seed {int(row['seed'])})",
                f"  final equity      ${row['final_equity']:,.2f}",
                f"  worst drawdown    {row['max_drawdown']:.1%}",
                f"  reached on        {row['date_to_target']}",
            ]
        else:
            best = self.attempts.sort_values("peak_equity", ascending=False).iloc[0]
            lines += [
                "  target NOT reached by any attempt",
                f"  best attempt      {best['label']} (seed {int(best['seed'])}) "
                f"peaked at ${best['peak_equity']:,.2f}",
            ]
        lines.append(
            "  note              every extra attempt is another chance to fit noise; "
            "read the attempt count\n                    as part of the result, and check the "
            "deflated Sharpe against it."
        )
        return "\n".join(lines)


def search_until_target(
    datasets: Dict[str, pd.DataFrame | Dict[str, pd.DataFrame]],
    base_exec: ExecutionConfig | None = None,
    wf: WalkForwardConfig | None = None,
    space: Dict[str, Sequence] | None = None,
    *,
    seeds: Sequence[int] = (1, 2, 3),
    stop_when_reached: bool = True,
    verbose: bool = True,
) -> TargetSearchResult:
    """Run walk-forwards across markets and seeds until the target is reached.

    This automates "keep backtesting until it makes $1,000" - and, because that
    loop is itself a search, it keeps the receipt. Every attempt is recorded, so
    the number of attempts can be fed to :func:`~tradingagent.metrics.deflated_sharpe`
    and the result read for what it is: the best of *n* tries, not one clean
    experiment. An attempt "reaches the target" only on its out-of-sample curve;
    the search never sees the windows it is judged on.
    """
    base_exec = base_exec or ExecutionConfig()
    wf = wf or WalkForwardConfig(verbose=False)
    target = base_exec.target_equity

    rows: List[dict] = []
    best_result: Optional[WalkForwardResult] = None
    best_label = ""
    best_peak = -np.inf
    reached = False

    for label, data in datasets.items():
        for seed in seeds:
            try:
                res = walk_forward(data, base_exec, replace(wf, seed=seed, verbose=False), space)
            except ValueError as exc:  # not enough history for this market
                if verbose:
                    print(f"[skip] {label} seed {seed}: {exc}")
                continue
            stats = res.stats()
            peak = float(res.equity.max())
            hit = bool(stats.get("target_hit", 0.0))
            rows.append(
                {
                    "label": label,
                    "seed": seed,
                    "final_equity": stats["final_equity"],
                    "peak_equity": peak,
                    "target_hit": hit,
                    "date_to_target": str(stats.get("date_to_target", ""))[:10],
                    "sharpe": stats["sharpe"],
                    "max_drawdown": stats["max_drawdown"],
                    "n_trades": stats.get("n_trades", 0),
                }
            )
            if verbose:
                mark = "HIT" if hit else "   "
                print(
                    f"[{mark}] {label:<22} seed {seed}: ${stats['final_equity']:>9,.0f}  "
                    f"peak ${peak:>9,.0f}  sharpe {stats['sharpe']:5.2f}  "
                    f"maxDD {stats['max_drawdown']:6.1%}"
                )
            if peak > best_peak:
                best_peak, best_result, best_label = peak, res, f"{label} (seed {seed})"
            if hit:
                reached = True
                best_result, best_label = res, f"{label} (seed {seed})"
                if stop_when_reached:
                    return TargetSearchResult(
                        True, pd.DataFrame(rows), best_result, best_label, target
                    )

    return TargetSearchResult(reached, pd.DataFrame(rows), best_result, best_label, target)
