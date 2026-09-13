"""Position sizing and capital protection.

Signals say *which way* to lean; this module decides *how much*. Three layers,
applied in order:

1. :func:`vol_target_scale` - size so that expected portfolio volatility is
   roughly constant, which stops a quiet regime from under-trading and a
   violent one from wiping the account.
2. :func:`kelly_fraction` - a causal, heavily-discounted Kelly estimate from
   trailing performance.
3. ATR stops and the drawdown kill switch, which the engine applies
   bar-by-bar because they depend on the realised equity path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class RiskConfig:
    """Knobs for the sizing layers."""

    target_vol: float = 0.50          # annualised portfolio vol aimed for
    vol_lookback: int = 30            # bars used to estimate realised vol
    max_leverage: float = 2.0         # hard cap on gross exposure / equity
    min_leverage: float = 0.0
    atr_stop_mult: float = 6.0        # 0 disables the stop. Wide on purpose: a
                                      # 2-3 ATR stop is inside daily crypto noise
                                      # and gets shaken out of the trends the
                                      # trend models exist to capture.
    atr_n: int = 14
    trail_stop: bool = True
    take_profit_mult: float = 0.0     # 0 disables
    reentry_lockout_bars: int = 3     # bars to stand aside after a stop-out
    max_drawdown_stop: float = 0.35   # flatten if equity falls this far from peak
    cooldown_bars: int = 10           # bars to stay flat after the kill switch
    kelly_lookback: int = 0           # 0 disables the Kelly layer. It stays off:
                                      # see kelly_fraction() for why the evidence
                                      # does not support sizing on it.
    kelly_fraction: float = 0.25      # fraction of full Kelly to actually use
    kelly_min_observations: int = 120 # refuse to size on a shorter sample
    kelly_cap: float = 1.0            # never exceed full Kelly
    risk_per_trade: float = 0.0       # if > 0, size so an ATR stop costs this
                                      # fraction of equity


def vol_target_scale(
    close: pd.Series,
    *,
    target_vol: float,
    lookback: int,
    periods_per_year: float,
    max_scale: float = 10.0,
) -> pd.Series:
    """Multiplier that pushes realised vol towards ``target_vol``."""
    realised = ind.realized_vol(close, lookback, periods_per_year)
    scale = target_vol / realised.replace(0.0, np.nan)
    # a fresh series has no vol estimate yet: stay small rather than guess
    return scale.clip(upper=max_scale).fillna(0.0)


def atr_risk_scale(
    df: pd.DataFrame, *, risk_per_trade: float, atr_mult: float, atr_n: int, max_scale: float = 10.0
) -> pd.Series:
    """Size so that hitting the ATR stop costs ``risk_per_trade`` of equity."""
    if risk_per_trade <= 0 or atr_mult <= 0:
        return pd.Series(1.0, index=df.index)
    a = ind.atr(df["high"], df["low"], df["close"], atr_n)
    stop_distance = (atr_mult * a / df["close"]).replace(0.0, np.nan)
    return (risk_per_trade / stop_distance).clip(upper=max_scale).fillna(0.0)


def kelly_fraction(
    strategy_returns: pd.Series,
    *,
    lookback: int,
    cap: float = 1.0,
    min_observations: int = 120,
    shrink: bool = True,
) -> pd.Series:
    """Shrunk, capped, sample-size-gated Kelly estimate.

    Kelly is ``mu / sigma^2``, and the honest problem with it is that both terms
    are estimated. ``mu`` is the hard one: its standard error is
    ``sigma / sqrt(n)``, so over 60 daily bars the estimate of the mean is
    dominated by noise, and dividing a noisy mean by a comparatively
    well-estimated variance converts that noise directly into leverage. The
    previous implementation accepted as few as ``lookback // 2`` observations
    and allowed a 4x multiplier, which is how a run of luck became a large
    position.

    Three guards, all of which cost return in a backtest and are worth it:

    * **A sample-size floor.** Nothing is sized until ``min_observations`` bars
      exist. Below that the function returns zero, not a guess.
    * **Confidence shrinkage.** The estimate is scaled by
      ``1 - 1/t²`` where ``t`` is the trailing mean's t-statistic, so an edge
      indistinguishable from zero produces no position and one that is strongly
      established produces nearly the full estimate.
    * **A cap of 1.0 by default.** Above full Kelly, expected growth *falls*
      while variance keeps rising; there is never a reason to be there.

    It remains **disabled by default** (``RiskConfig.kelly_lookback = 0``). The
    evidence in this repository is not sufficient to justify sizing on it, and
    the right response to insufficient evidence is to leave the component off
    rather than to delete it.
    """
    if lookback <= 0:
        return pd.Series(1.0, index=strategy_returns.index)

    window = max(int(lookback), int(min_observations))
    mean = strategy_returns.rolling(window, min_periods=window).mean()
    std = strategy_returns.rolling(window, min_periods=window).std(ddof=1)
    var = std**2
    estimate = mean / var.replace(0.0, np.nan)

    if shrink:
        # t-statistic of the trailing mean; below |t| = 1 the edge is noise
        t_stat = mean / (std.replace(0.0, np.nan) / np.sqrt(window))
        confidence = (1.0 - 1.0 / t_stat.pow(2).clip(lower=1e-9)).clip(lower=0.0, upper=1.0)
        estimate = estimate * confidence

    # shift so today's size cannot use today's realised return
    return estimate.shift(1).clip(lower=0.0, upper=cap).fillna(0.0)


def apply_sizing(
    df: pd.DataFrame,
    weight: pd.Series,
    cfg: RiskConfig,
    *,
    periods_per_year: float = 365.0,
    strategy_returns: pd.Series | None = None,
) -> pd.Series:
    """Turn a raw signal in ``[-1, 1]`` into a leveraged target weight."""
    w = weight.astype(float).fillna(0.0)

    scale = pd.Series(1.0, index=w.index)
    if cfg.target_vol > 0:
        scale = scale * vol_target_scale(
            df["close"],
            target_vol=cfg.target_vol,
            lookback=cfg.vol_lookback,
            periods_per_year=periods_per_year,
        )
    if cfg.risk_per_trade > 0:
        scale = scale * atr_risk_scale(
            df,
            risk_per_trade=cfg.risk_per_trade,
            atr_mult=cfg.atr_stop_mult,
            atr_n=cfg.atr_n,
        )
    if cfg.kelly_lookback > 0 and strategy_returns is not None:
        scale = scale * kelly_fraction(
            strategy_returns,
            lookback=cfg.kelly_lookback,
            cap=cfg.kelly_cap,
            min_observations=cfg.kelly_min_observations,
        ) * cfg.kelly_fraction

    sized = (w * scale).clip(-cfg.max_leverage, cfg.max_leverage)
    small = sized.abs() < cfg.min_leverage
    sized = sized.where(~small, np.sign(sized) * cfg.min_leverage)
    return sized.fillna(0.0)
