"""Cross-sectional features.

Every function here takes and returns *wide* frames (dates down, symbols
across), so a cross-sectional operation is one row-wise call.

Two separate causality rules apply, and it is worth being precise about which
is which:

* **Down the time axis** - a feature at date ``t`` may use bars at or before
  ``t`` and nothing after. Same rule as everywhere else in the package.
* **Across the symbol axis** - standardising a feature at date ``t`` uses every
  symbol's value *on that same date*. That is not lookahead: at the close of
  ``t`` you can see every name's price. It does assume you could have computed
  the cross-section in time to trade the next open, which for daily bars and a
  hundred liquid names is true.

The target is a forward return, so fitting needs care: see
:func:`purge_overlapping`, which drops the training rows whose outcome had not
finished happening by the end of the training window.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .universe import Panel

#: Default feature set. Names are the model's coefficient labels, so keep them
#: short and readable - they end up in the stability report.
DEFAULT_FEATURES: List[str] = [
    "mom_12_1",
    "mom_6_1",
    "mom_3",
    "accel",
    "rev_5",
    "rev_21",
    "vol_63",
    "mom_risk_adj",
    "dist_200",
    "rsi_14",
    "liquidity",
]


# --------------------------------------------------------------------------- #
# raw features
# --------------------------------------------------------------------------- #
def _returns(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close).diff()


def raw_features(panel: Panel, *, periods_per_year: float = 252.0) -> Dict[str, pd.DataFrame]:
    """Per-name features, before any cross-sectional treatment."""
    close, volume = panel.close, panel.volume
    rets = _returns(close)

    vol_63 = rets.rolling(63, min_periods=40).std(ddof=0) * np.sqrt(periods_per_year)
    vol_126 = rets.rolling(126, min_periods=80).std(ddof=0) * np.sqrt(periods_per_year)

    # 12-1 momentum: the classic - a year of trend, with the most recent month
    # left out because short-horizon reversal runs the other way
    mom_12_1 = close.shift(21) / close.shift(252) - 1.0
    mom_6_1 = close.shift(21) / close.shift(126) - 1.0
    mom_3 = close / close.shift(63) - 1.0

    feats: Dict[str, pd.DataFrame] = {
        "mom_12_1": mom_12_1,
        "mom_6_1": mom_6_1,
        "mom_3": mom_3,
        "accel": mom_3 - mom_6_1,
        "rev_5": -(close / close.shift(5) - 1.0),
        "rev_21": -(close / close.shift(21) - 1.0),
        "vol_63": vol_63,
        "mom_risk_adj": (close / close.shift(126) - 1.0) / vol_126.replace(0.0, np.nan),
        "dist_200": close / close.rolling(200, min_periods=120).mean() - 1.0,
        "rsi_14": _rsi(close, 14),
        "liquidity": np.log1p((close * volume).rolling(21, min_periods=10).mean()),
    }
    return feats


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# --------------------------------------------------------------------------- #
# cross-sectional treatment
# --------------------------------------------------------------------------- #
def soft_clip(z: pd.DataFrame, limit: float) -> pd.DataFrame:
    """Compress the tails beyond ``limit`` without collapsing them to one value.

    A hard ``clip`` sets every extreme name to exactly the limit, so they tie -
    and a ranking model then picks between tied names by column order rather
    than by signal. In one run three of twelve holdings were tied at the clip.
    This keeps the outlier's influence on the scale bounded while staying
    strictly monotone, so the ordering survives: values inside the limit are
    untouched, and everything beyond it is squashed into ``(limit, limit + 1)``.
    """
    magnitude = z.abs()
    excess = (magnitude - limit).clip(lower=0.0)
    return np.sign(z) * (magnitude.clip(upper=limit) + np.tanh(excess))


def cross_sectional_z(
    frame: pd.DataFrame, *, clip: float = 3.0, min_names: int = 5
) -> pd.DataFrame:
    """Standardise each row across symbols, ignoring missing names.

    Taming the tails first stops one blown-up name from setting the scale for
    the whole row, which would shrink every other name's score toward zero.
    """
    valid = frame.notna().sum(axis=1)
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = frame.sub(mean, axis=0).div(std, axis=0)
    z = soft_clip(z, clip)
    # re-standardise afterwards so the row is still mean 0, unit scale
    z = z.sub(z.mean(axis=1), axis=0).div(z.std(axis=1, ddof=0).replace(0.0, np.nan), axis=0)
    return z.where(valid.ge(min_names), np.nan)


def feature_panel(
    panel: Panel,
    features: Sequence[str] | None = None,
    *,
    periods_per_year: float = 252.0,
    clip: float = 3.0,
    min_names: int = 5,
) -> Dict[str, pd.DataFrame]:
    """Standardised feature frames, ready to be scored or fitted."""
    names = list(features or DEFAULT_FEATURES)
    raw = raw_features(panel, periods_per_year=periods_per_year)
    unknown = [n for n in names if n not in raw]
    if unknown:
        raise KeyError(f"unknown features: {unknown}; available: {sorted(raw)}")
    tradeable = panel.tradeable()
    return {
        name: cross_sectional_z(raw[name].where(tradeable), clip=clip, min_names=min_names)
        for name in names
    }


def forward_return(
    close: pd.DataFrame, horizon: int = 21, *, demean: bool = True
) -> pd.DataFrame:
    """Return over the next ``horizon`` bars, optionally cross-sectionally demeaned.

    Demeaning is what turns "will this go up" into "will this beat its peers",
    which is the question a ranking model can actually answer - it strips out
    the market move that every name shares.
    """
    fwd = close.shift(-horizon) / close - 1.0
    if demean:
        fwd = fwd.sub(fwd.mean(axis=1), axis=0)
    return fwd


# --------------------------------------------------------------------------- #
# stacking for model fitting
# --------------------------------------------------------------------------- #
def stack(
    features: Dict[str, pd.DataFrame], target: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], pd.MultiIndex, List[str]]:
    """Flatten wide frames into the ``(n_observations, n_features)`` matrix.

    Rows with any missing feature - or a missing target, when one is given - are
    dropped, so the model never sees a half-formed observation.
    """
    names = list(features)
    long = pd.concat([features[n].stack(future_stack=True).rename(n) for n in names], axis=1)
    if target is not None:
        long["__y__"] = target.stack(future_stack=True)
    long = long.dropna()
    y = long.pop("__y__").to_numpy() if target is not None else None
    return long[names].to_numpy(), y, long.index, names


def purge_overlapping(index: pd.MultiIndex, train_end: pd.Timestamp, horizon: int, dates: pd.DatetimeIndex) -> np.ndarray:
    """Mask of training rows whose outcome finished inside the training window.

    A target that is a 21-bar forward return observed on the last day of the
    training window resolves 21 bars *into the test window*. Fitting on it leaks
    the test period into the model. This drops those rows.
    """
    obs_dates = index.get_level_values(0)
    cutoff_pos = dates.searchsorted(train_end) - horizon
    if cutoff_pos < 0:
        return np.zeros(len(index), dtype=bool)
    cutoff = dates[max(cutoff_pos, 0)]
    return np.asarray(obs_dates <= cutoff)
