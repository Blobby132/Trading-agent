"""Cross-sectional ranking: score every name against its peers, trade the spread.

The single-asset agent asks "is this going up?". That question has a terrible
signal-to-noise ratio, because most of any one stock's move is the market's move.
This module asks a different one: **"which of these names will beat the others?"**
The market component cancels, and a hundred names give a hundred comparisons a
day instead of one forecast.

Three rankers, in increasing order of how much they learn:

* :class:`SingleFeatureRanker` - one factor, no fitting. The baseline any
  learned model has to beat before it has earned its complexity.
* :class:`EqualBlendRanker` - a fixed, signed blend of standardised features.
  Still no fitting, but it is a portfolio of ideas rather than one idea.
* :class:`RidgeRanker` - learns the feature weights from the training window,
  regularised hard because the whole difficulty here is that the signal is small
  and the estimation error is not.

The walk-forward loop then does the selecting: fit every candidate on the
training window, trade the best few on the window that follows, and repeat.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import DEFAULT_FEATURES, feature_panel, forward_return, purge_overlapping, stack
from .universe import Panel

#: Which way each standardised feature is expected to point, used by the
#: no-fitting baselines. Note `rev_*` features are already defined as the
#: negative of a return, so +1 there means "fade the recent move".
DEFAULT_SIGNS: Dict[str, float] = {
    "mom_12_1": 1.0,
    "mom_6_1": 1.0,
    "mom_3": 1.0,
    "accel": 1.0,
    "rev_5": 1.0,
    "rev_21": 1.0,
    "vol_63": -1.0,
    "mom_risk_adj": 1.0,
    "dist_200": 1.0,
    "rsi_14": 0.0,
    "liquidity": 0.0,
}


# --------------------------------------------------------------------------- #
# rankers
# --------------------------------------------------------------------------- #
class Ranker(ABC):
    """Maps a feature panel to a per-date, per-name score."""

    name: str = "ranker"

    @abstractmethod
    def fit(
        self,
        features: Dict[str, pd.DataFrame],
        target: pd.DataFrame,
        *,
        train_end: pd.Timestamp,
        train_start: pd.Timestamp,
        horizon: int,
    ) -> "Ranker":
        ...

    @abstractmethod
    def score(self, features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        ...

    def coefficients(self) -> Dict[str, float]:
        return {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.name})"


def _blend(features: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.DataFrame:
    """Weighted sum of standardised feature frames, ignoring missing values."""
    used = {k: w for k, w in weights.items() if abs(w) > 1e-12 and k in features}
    if not used:
        any_frame = next(iter(features.values()))
        return pd.DataFrame(np.nan, index=any_frame.index, columns=any_frame.columns)
    total = None
    count = None
    for key, weight in used.items():
        frame = features[key]
        contrib = frame.fillna(0.0) * weight
        present = frame.notna().astype(float) * abs(weight)
        total = contrib if total is None else total + contrib
        count = present if count is None else count + present
    score = total / count.replace(0.0, np.nan)
    return score.where(count > 0)


class SingleFeatureRanker(Ranker):
    """One factor, no fitting - the baseline to beat."""

    def __init__(self, feature: str = "mom_12_1", sign: float = 1.0):
        self.feature, self.sign = feature, sign
        self.name = f"single:{feature}"

    def fit(self, features, target, **kwargs) -> "SingleFeatureRanker":
        return self

    def score(self, features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return _blend(features, {self.feature: self.sign})

    def coefficients(self) -> Dict[str, float]:
        return {self.feature: self.sign}


class EqualBlendRanker(Ranker):
    """Fixed, signed blend of every feature - a portfolio of ideas, still unfitted."""

    def __init__(self, signs: Dict[str, float] | None = None):
        self.signs = dict(signs or DEFAULT_SIGNS)
        self.name = "equal_blend"

    def fit(self, features, target, **kwargs) -> "EqualBlendRanker":
        return self

    def score(self, features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return _blend(features, self.signs)

    def coefficients(self) -> Dict[str, float]:
        return dict(self.signs)


class RidgeRanker(Ranker):
    """Ridge regression of forward relative return on standardised features.

    Solved in closed form - no scikit-learn dependency, and with standardised
    inputs there is nothing an optimiser would add.

    ``alpha`` is *relative*: the penalty is scaled by ``trace(X'X) / k``, so the
    same value means the same amount of shrinkage whether the training window
    holds two thousand observations or twenty thousand. Without that, a fixed
    alpha silently becomes weaker as the window grows.
    """

    def __init__(self, alpha: float = 10.0, fit_intercept: bool = False):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.name = f"ridge:{alpha:g}"
        self.coef_: Dict[str, float] = {}
        self.n_obs_: int = 0
        self.r2_: float = float("nan")

    def fit(
        self,
        features: Dict[str, pd.DataFrame],
        target: pd.DataFrame,
        *,
        train_end: pd.Timestamp,
        train_start: pd.Timestamp,
        horizon: int,
    ) -> "RidgeRanker":
        dates = next(iter(features.values())).index
        window = {k: v.loc[train_start:train_end] for k, v in features.items()}
        y_window = target.loc[train_start:train_end]
        X, y, idx, names = stack(window, y_window)
        if len(y) == 0:
            self.coef_ = {n: 0.0 for n in names}
            return self

        # drop observations whose forward return had not finished by train_end
        keep = purge_overlapping(idx, train_end, horizon, dates)
        return self.fit_matrix(X[keep], y[keep], names)

    def fit_matrix(self, X: np.ndarray, y: np.ndarray, names: Sequence[str]) -> "RidgeRanker":
        """Fit from an already-stacked design matrix.

        The walk-forward loop stacks the whole history once and slices rows per
        fold, which is far cheaper than re-stacking a wide panel for every
        candidate on every fold.
        """
        names = list(names)
        self.n_obs_ = int(len(y))
        if self.n_obs_ < 50:
            self.coef_ = {n: 0.0 for n in names}
            return self

        if self.fit_intercept:
            X = np.column_stack([X, np.ones(len(X))])
        gram = X.T @ X
        scale = np.trace(gram) / max(gram.shape[0], 1)
        ridge = self.alpha * scale * np.eye(gram.shape[0])
        try:
            beta = np.linalg.solve(gram + ridge, X.T @ y)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            beta = np.linalg.lstsq(gram + ridge, X.T @ y, rcond=None)[0]

        resid = y - X @ beta
        var = float(np.var(y))
        self.r2_ = float(1.0 - np.var(resid) / var) if var > 0 else float("nan")
        self.coef_ = {n: float(b) for n, b in zip(names, beta[: len(names)])}
        return self

    def score(self, features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not self.coef_:
            raise RuntimeError("RidgeRanker.score called before fit")
        return _blend(features, self.coef_)

    def coefficients(self) -> Dict[str, float]:
        return dict(self.coef_)


def make_ranker(spec: str, **params) -> Ranker:
    if spec == "ridge":
        return RidgeRanker(alpha=float(params.get("ridge_alpha", 10.0)))
    if spec == "equal_blend":
        return EqualBlendRanker()
    if spec == "mom_only":
        return SingleFeatureRanker("mom_12_1", 1.0)
    if spec == "rev_only":
        return SingleFeatureRanker("rev_21", 1.0)
    raise KeyError(f"unknown ranker {spec!r}")


# --------------------------------------------------------------------------- #
# scores -> portfolio weights
# --------------------------------------------------------------------------- #
@dataclass
class PortfolioRules:
    """How a score becomes a position."""

    long_only: bool = True
    top_frac: float = 0.20          # fraction of the live universe held long
    bottom_frac: float = 0.20       # fraction shorted (long/short only)
    gross: float = 1.0              # gross exposure as a multiple of equity
    max_weight: float = 0.15        # per-name cap
    min_names: int = 12             # below this the cross-section is too thin to rank
    min_positions: int = 5          # never concentrate into one or two names
    rebalance_every: int = 5        # bars between rebalances
    weighting: str = "equal"        # "equal" | "score"


def scores_to_weights(scores: pd.DataFrame, rules: PortfolioRules) -> pd.DataFrame:
    """Turn per-name scores into per-name portfolio weights."""
    valid = scores.notna()
    n_live = valid.sum(axis=1)

    desc = scores.rank(axis=1, ascending=False, method="first")
    asc = scores.rank(axis=1, ascending=True, method="first")
    # A fraction of a thin universe is one or two names, which is a coin flip
    # rather than a cross-section. Two floors apply per side: `min_positions`,
    # and however many names the per-name cap needs to fund the gross budget
    # (a 1.0 book capped at 0.1 per name needs at least ten). The ceiling is
    # half the live universe, so the long and short sides cannot overlap.
    budget = rules.gross if rules.long_only else rules.gross / 2.0
    needed = int(np.ceil(budget / max(rules.max_weight, 1e-9)))
    floor = max(int(rules.min_positions), needed)
    half = (n_live / 2).clip(lower=1)
    n_long = (rules.top_frac * n_live).round().clip(lower=floor).clip(upper=half)
    n_short = (rules.bottom_frac * n_live).round().clip(lower=floor).clip(upper=half)

    long_mask = desc.le(n_long, axis=0) & valid
    short_mask = (asc.le(n_short, axis=0) & valid) if not rules.long_only else valid & False

    if rules.weighting == "score":
        centred = scores.sub(scores.mean(axis=1), axis=0)
        long_raw = centred.where(long_mask).clip(lower=0.0).fillna(0.0)
        short_raw = (-centred).where(short_mask).clip(lower=0.0).fillna(0.0)
    else:
        long_raw = long_mask.astype(float)
        short_raw = short_mask.astype(float)

    def _norm(raw: pd.DataFrame, budget: float) -> pd.DataFrame:
        total = raw.sum(axis=1).replace(0.0, np.nan)
        return raw.div(total, axis=0).fillna(0.0) * budget

    if rules.long_only:
        weights = _norm(long_raw, rules.gross)
    else:
        half = rules.gross / 2.0
        weights = _norm(long_raw, half) - _norm(short_raw, half)

    # The cap is a hard limit, so it is applied last and nothing is scaled back
    # up afterwards: if the universe is too thin to fund the full gross budget
    # within the cap, the book simply runs smaller. That is the safe direction.
    weights = weights.clip(-rules.max_weight, rules.max_weight)

    # too few names to rank meaningfully: stand aside
    weights = weights.where(n_live.ge(rules.min_names), 0.0)

    if rules.rebalance_every > 1:
        # hold the book between rebalance dates; turnover is the main cost here
        keep = np.zeros(len(weights), dtype=bool)
        keep[:: int(rules.rebalance_every)] = True
        weights = weights.where(pd.Series(keep, index=weights.index), np.nan).ffill()
    return weights.fillna(0.0)


def volatility_target(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    *,
    target_vol: float,
    lookback: int = 63,
    periods_per_year: float = 252.0,
    max_scale: float = 3.0,
) -> pd.DataFrame:
    """Scale the whole book so its realised volatility sits near ``target_vol``.

    The volatility estimate is built from the portfolio's own hypothetical
    returns, lagged one bar, so the size used today never depends on today's
    outcome.
    """
    if target_vol <= 0:
        return weights
    asset_ret = close.pct_change()
    port_ret = (weights.shift(1) * asset_ret).sum(axis=1)
    realised = port_ret.rolling(lookback, min_periods=lookback // 2).std(ddof=0) * np.sqrt(
        periods_per_year
    )
    scale = (target_vol / realised.replace(0.0, np.nan)).shift(1).clip(upper=max_scale)
    return weights.mul(scale.fillna(0.0), axis=0)
