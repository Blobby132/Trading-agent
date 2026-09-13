"""Walk-forward learning for the cross-sectional agent.

This is the loop the whole idea rests on: on each training window, fit every
candidate model, score them, trade the best few on the window that follows, and
repeat. The model's *coefficients* are refitted every fold, and so is the
*choice* of which model to use - that is what makes it adaptive rather than a
fixed rule with a fancy name.

What it deliberately does not do is let anything about the test window influence
what gets traded in it. Three separate guards:

* the fit only ever sees bars inside the training window;
* observations whose forward return had not finished by the end of that window
  are purged, because their outcome lies in the test period;
* an embargo of further bars separates the two, so rolling features cannot
  smear across the join.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .cross_section import (
    PortfolioRules,
    Ranker,
    RidgeRanker,
    make_ranker,
    scores_to_weights,
    volatility_target,
)
from .engine import BacktestEngine, BacktestResult, ExecutionConfig
from .execution import BASE_COST, CostModel
from .features import DEFAULT_FEATURES, feature_panel, forward_return
from .holdout import Holdout
from .ledger import ResearchLedger
from .metrics import summarize
from .optimize import OBJECTIVES, sample_unique
from .risk import RiskConfig
from .universe import Panel

#: What the optimiser is allowed to choose between on each training window.
XS_SEARCH_SPACE: Dict[str, Sequence] = {
    "ranker": ["ridge", "equal_blend", "mom_only", "rev_only"],
    "ridge_alpha": [1.0, 10.0, 50.0, 200.0],
    "horizon": [5, 21, 63],
    "long_only": [0, 1],
    "top_frac": [0.10, 0.20, 0.30],
    "rebalance_every": [1, 5, 21],
    "max_weight": [0.10, 0.20],
    "gross": [1.0, 1.5],
    "weighting": ["equal", "score"],
    "vol_target": [0.0, 0.15, 0.25],
}

#: A long-only variant, for accounts that cannot short (which includes any small
#: US cash account - shorting needs margin, and Reg T sets a $2,000 floor).
XS_SEARCH_SPACE_LONG_ONLY: Dict[str, Sequence] = {**XS_SEARCH_SPACE, "long_only": [1]}


def params_to_rules(params: Dict) -> PortfolioRules:
    return PortfolioRules(
        long_only=bool(int(params["long_only"])),
        top_frac=float(params["top_frac"]),
        bottom_frac=float(params["top_frac"]),
        gross=float(params["gross"]),
        max_weight=float(params["max_weight"]),
        rebalance_every=int(params["rebalance_every"]),
        weighting=str(params["weighting"]),
    )


@dataclass
class XSWalkForwardConfig:
    train_bars: int = 756          # ~3 years of trading days
    test_bars: int = 126           # ~6 months traded out of sample
    embargo_bars: int = 21         # gap between fit and trade
    n_candidates: int = 60
    top_k: int = 3
    objective: str = "target_growth"
    max_train_drawdown: float = 0.45
    seed: int = 0
    verbose: bool = True


@dataclass
class XSWalkForwardResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    folds: pd.DataFrame
    coefficients: pd.DataFrame       # per fold, the winning model's feature weights
    chosen: List[List[Dict]]
    exec_config: ExecutionConfig
    risk_config: RiskConfig
    trades: pd.DataFrame
    costs: pd.DataFrame
    n_evaluations: int = 0
    meta: Dict[str, object] = field(default_factory=dict)

    def stats(self, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
        result = BacktestResult(
            equity=self.equity,
            returns=self.returns,
            weights=self.weights,
            target_weights=self.weights,
            trades=self.trades,
            costs=self.costs,
            exec_config=self.exec_config,
            risk_config=self.risk_config,
            meta={"bust": bool(self.equity.iloc[-1] <= self.exec_config.ruin_equity)},
        )
        return summarize(result, benchmark=benchmark)

    def coefficient_stability(self) -> Dict[str, float]:
        """How much the learned model moves between folds.

        A model whose coefficient vector points somewhere new every six months
        is fitting noise. The mean correlation between consecutive folds' vectors
        is the cheapest available check on that; ``mean_abs_coef`` is reported
        alongside it because a stable vector of near-zeros is stable and useless.
        """
        # folds won by a model without coefficients (the unfitted baselines)
        # contribute nothing to a stability measure of the *learned* weights
        coefs = self.coefficients.dropna(how="all").fillna(0.0)
        if len(coefs) < 2:
            return {"n_folds": float(len(coefs))}
        vals = coefs.to_numpy()
        corrs = [
            float(np.corrcoef(vals[i], vals[i + 1])[0, 1])
            for i in range(len(vals) - 1)
            if np.std(vals[i]) > 0 and np.std(vals[i + 1]) > 0
        ]
        signs = (coefs > 0).mean(axis=0)
        return {
            "n_folds": float(len(coefs)),
            "mean_consecutive_correlation": float(np.mean(corrs)) if corrs else float("nan"),
            "min_consecutive_correlation": float(np.min(corrs)) if corrs else float("nan"),
            "mean_abs_coef": float(np.abs(vals).mean()),
            "most_stable_sign": float(np.max(np.abs(signs - 0.5)) * 2.0),
        }


# --------------------------------------------------------------------------- #
def _stacked_design(features: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, pd.MultiIndex, List[str]]:
    names = list(features)
    wide = pd.concat(
        [features[n].stack(future_stack=True).rename(n) for n in names], axis=1
    ).dropna()
    return wide[names].to_numpy(), wide.index, names


def walk_forward_xs(
    panel: Panel,
    base_exec: ExecutionConfig | None = None,
    wf: XSWalkForwardConfig | None = None,
    space: Dict[str, Sequence] | None = None,
    *,
    features: Sequence[str] | None = None,
    risk: RiskConfig | None = None,
    holdout: Optional["Holdout"] = None,
    ledger: Optional["ResearchLedger"] = None,
    label: str = "walk_forward_xs",
) -> XSWalkForwardResult:
    """Fit, select, trade, step forward - and report only the traded windows.

    Pass ``holdout`` to make the reserved era unreachable, and ``ledger`` to
    record the search so repeated experimentation stays auditable.
    """
    base_exec = base_exec or ExecutionConfig(periods_per_year=252.0)
    wf = wf or XSWalkForwardConfig()
    space = space or XS_SEARCH_SPACE
    risk = risk or RiskConfig(
        target_vol=0.0, atr_stop_mult=0.0, max_drawdown_stop=0.0, reentry_lockout_bars=0
    )
    objective = OBJECTIVES[wf.objective]
    feature_names = list(features or DEFAULT_FEATURES)
    if holdout is not None:
        holdout.guard(panel, what="walk_forward_xs")

    feats = feature_panel(panel, feature_names, periods_per_year=base_exec.periods_per_year)
    dates = panel.index
    frames = panel.to_frames()

    # stack once; per fold we only mask rows, which is far cheaper
    X_all, idx_all, names = _stacked_design(feats)
    obs_dates = idx_all.get_level_values(0)
    horizons = sorted({int(h) for h in space["horizon"]})
    targets = {h: forward_return(panel.close, h, demean=True) for h in horizons}
    y_all = {
        h: targets[h].stack(future_stack=True).reindex(idx_all).to_numpy() for h in horizons
    }

    candidates = sample_unique(wf.n_candidates, space, seed=wf.seed)
    n = len(dates)

    equity_pieces: List[pd.Series] = []
    weight_pieces: List[pd.DataFrame] = []
    trade_pieces: List[pd.DataFrame] = []
    cost_pieces: List[pd.DataFrame] = []
    fold_rows: List[dict] = []
    coef_rows: List[dict] = []
    chosen_per_fold: List[List[Dict]] = []

    capital = float(base_exec.initial_capital)
    start = 0
    fold_id = 0
    evaluations = 0

    while start + wf.train_bars + wf.embargo_bars + 1 < n:
        train = slice(start, start + wf.train_bars)
        test_start = start + wf.train_bars + wf.embargo_bars
        test_end = min(test_start + wf.test_bars, n)
        if test_end - test_start < 5:
            break
        test = slice(test_start, test_end)
        train_start_ts, train_end_ts = dates[train.start], dates[train.stop - 1]

        scored: List[Tuple[float, int, pd.DataFrame, Dict[str, float]]] = []
        for cand_id, params in enumerate(candidates):
            horizon = int(params["horizon"])
            ranker = make_ranker(str(params["ranker"]), **params)
            if isinstance(ranker, RidgeRanker):
                y = y_all[horizon]
                # rows inside the training window whose outcome also resolved inside it
                cutoff_pos = max(dates.searchsorted(train_end_ts) - horizon, 0)
                cutoff = dates[cutoff_pos]
                mask = (
                    np.asarray(obs_dates >= train_start_ts)
                    & np.asarray(obs_dates <= cutoff)
                    & np.isfinite(y)
                )
                ranker.fit_matrix(X_all[mask], y[mask], names)

            scores = ranker.score(feats)
            weights = scores_to_weights(scores, params_to_rules(params))
            if float(params["vol_target"]) > 0:
                weights = volatility_target(
                    weights,
                    panel.close,
                    target_vol=float(params["vol_target"]),
                    periods_per_year=base_exec.periods_per_year,
                )

            train_res = _run(
                frames, weights, train, base_exec, risk, base_exec.initial_capital, record=False
            )
            stats = summarize(train_res)
            score = objective(stats)
            if abs(stats.get("max_drawdown", -1.0)) > wf.max_train_drawdown:
                score -= 1000.0
            scored.append((score, cand_id, weights, ranker.coefficients()))
            evaluations += 1

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[: max(1, wf.top_k)]
        chosen_per_fold.append([candidates[t[1]] for t in top])

        blended = sum(t[2] for t in top) / len(top)
        exec_cfg = replace(base_exec, max_leverage=max(base_exec.max_leverage, 1.0))
        res = _run(frames, blended, test, exec_cfg, risk, capital)

        equity_pieces.append(res.equity)
        # the realised book, not the requested one: the engine caps gross
        # exposure and skips untradeable names, so storing the request would
        # make every exposure statistic describe an intention instead of a position
        weight_pieces.append(res.weights)
        cost_pieces.append(res.costs)
        if not res.trades.empty:
            trade_pieces.append(res.trades)

        best_params = candidates[top[0][1]]
        # Normalise each fold's coefficients to unit length before recording.
        # A ridge vector predicts raw returns and lives at ~1e-3, while the
        # unfitted baselines use +/-1 signs; without this the two are averaged
        # on incomparable scales and the report is meaningless. Ranking only
        # depends on direction, so the normalisation loses nothing.
        coef_rows.append({"fold": fold_id, **_unit_norm(top[0][3])})
        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": train_start_ts,
                "train_end": train_end_ts,
                "test_start": dates[test_start],
                "test_end": dates[test_end - 1],
                "start_equity": capital,
                "end_equity": float(res.equity.iloc[-1]),
                "return": float(res.equity.iloc[-1] / capital - 1.0),
                "max_drawdown": float((res.equity / res.equity.cummax() - 1.0).min()),
                "trades": int(len(res.trades)),
                "model": best_params["ranker"],
                "horizon": best_params["horizon"],
                "long_only": bool(int(best_params["long_only"])),
                "top_frac": best_params["top_frac"],
                "rebalance_every": best_params["rebalance_every"],
                "train_score": float(top[0][0]),
            }
        )
        if wf.verbose:
            row = fold_rows[-1]
            print(
                f"[fold {fold_id:>2}] {row['test_start'].date()} -> {row['test_end'].date()}  "
                f"${row['start_equity']:>9,.2f} -> ${row['end_equity']:>9,.2f} "
                f"({row['return']:+7.1%})  model={row['model']}/h{row['horizon']}"
                f"{'/LO' if row['long_only'] else '/LS'}"
            )

        capital = max(float(res.equity.iloc[-1]), 0.0)
        if capital <= base_exec.ruin_equity:
            if wf.verbose:
                print(f"[fold {fold_id}] account ruined - stopping")
            break
        start += wf.test_bars
        fold_id += 1

    if not equity_pieces:
        raise ValueError(
            f"not enough data for a walk-forward: {n} bars, needs more than "
            f"{wf.train_bars + wf.embargo_bars + wf.test_bars}"
        )

    equity = pd.concat(equity_pieces)
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    weights = pd.concat(weight_pieces)
    weights = weights[~weights.index.duplicated(keep="last")].sort_index()

    result = XSWalkForwardResult(
        equity=equity.rename("equity"),
        returns=equity.pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0),
        weights=weights,
        folds=pd.DataFrame(fold_rows),
        coefficients=pd.DataFrame(coef_rows).set_index("fold") if coef_rows else pd.DataFrame(),
        chosen=chosen_per_fold,
        exec_config=base_exec,
        risk_config=risk,
        trades=pd.concat(trade_pieces) if trade_pieces else pd.DataFrame(),
        costs=pd.concat(cost_pieces) if cost_pieces else pd.DataFrame(index=equity.index),
        n_evaluations=evaluations,
        meta={"n_candidates": len(candidates), "features": feature_names,
              "symbols": panel.symbols},
    )
    if ledger is not None:
        ledger.record(
            label, result.stats(), universe=f"{len(panel.symbols)}_names",
            n_symbols=len(panel.symbols), dataset_start=dates[0], dataset_end=dates[-1],
            candidate_count=evaluations,
            parameters=dict(chosen_per_fold[-1][0]) if chosen_per_fold else {},
            seed=wf.seed, holdout_status="holdout" if holdout is None else "development",
            objective=wf.objective, cost_scenario=base_exec.cost_model().name,
            used_for_selection=True, notes=f"{len(fold_rows)} folds, top_k={wf.top_k}",
        )
    return result


def _unit_norm(coefficients: Dict[str, float]) -> Dict[str, float]:
    """Scale a coefficient vector to unit length, leaving an all-zero one alone."""
    if not coefficients:
        return {}
    norm = float(np.sqrt(sum(v * v for v in coefficients.values())))
    if norm <= 0:
        return dict(coefficients)
    return {k: float(v) / norm for k, v in coefficients.items()}


def _run(frames, weights, window: slice, exec_cfg, risk, capital, *, record: bool = True) -> BacktestResult:
    sliced = {k: v.iloc[window] for k, v in frames.items()}
    cfg = replace(exec_cfg, initial_capital=float(capital), record_trades=record)
    return BacktestEngine(cfg, risk).run(sliced, weights.iloc[window])


def equal_weight_benchmark(
    panel: Panel,
    initial_capital: float = 100.0,
    *,
    rebalance_every: int = 21,
    costs: Optional["CostModel"] = None,
    periods_per_year: float = 252.0,
) -> pd.Series:
    """Buy the whole universe in equal weights - the benchmark that matters.

    Beating one stock is luck. Beating an equal-weight basket of the same names
    you were choosing from is the only comparison that says the *ranking* added
    anything.

    The benchmark is charged **the same costs as the strategy**. It previously
    paid 1 bp of fee and 1 bp of slippage while strategies paid 15, which made
    it an easier target for no stated reason. Pass ``costs`` to match whatever
    scenario the strategy is being run under.
    """
    tradeable = panel.tradeable()
    weights = tradeable.astype(float)
    weights = weights.div(weights.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    if rebalance_every > 1:
        keep = np.zeros(len(weights), dtype=bool)
        keep[::rebalance_every] = True
        weights = weights.where(pd.Series(keep, index=weights.index), np.nan).ffill().fillna(0.0)
    cfg = ExecutionConfig(
        initial_capital=initial_capital, costs=costs or BASE_COST,
        min_trade_frac=0.0, max_leverage=1.0, periods_per_year=periods_per_year,
    )
    risk = RiskConfig(target_vol=0.0, atr_stop_mult=0.0, max_drawdown_stop=0.0)
    return BacktestEngine(cfg, risk).run(panel.to_frames(), weights).equity.rename("equal_weight")
