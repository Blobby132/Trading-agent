"""Performance statistics, including the ones this project actually cares about:
did the account reach the target, how long did it take, and how likely was that
to be luck.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: float) -> float:
    n = len(equity)
    if n < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    years = (n - 1) / periods_per_year
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def ann_volatility(returns: pd.Series, periods_per_year: float) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def _degenerate(sd: float, scale: float) -> bool:
    """True when a standard deviation is zero to within floating-point noise.

    An exact ``== 0`` test is not enough: the sample deviation of a constant
    series comes back as ~1e-19 rather than 0, which would otherwise divide
    into a Sharpe ratio of 1e16.
    """
    return not np.isfinite(sd) or sd <= 1e-12 * max(scale, 1.0)


def sharpe(returns: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    excess = returns - rf / periods_per_year
    sd = float(excess.std(ddof=0))
    if _degenerate(sd, float(excess.abs().mean())):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    dd = float(downside.std(ddof=0)) if len(downside) else 0.0
    if _degenerate(dd, float(excess.abs().mean())):
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def calmar(equity: pd.Series, periods_per_year: float) -> float:
    mdd = abs(max_drawdown(equity))
    g = cagr(equity, periods_per_year)
    if mdd == 0 or np.isnan(mdd) or np.isnan(g):
        return float("nan")
    return float(g / mdd)


def ulcer_index(equity: pd.Series) -> float:
    dd = (equity / equity.cummax() - 1.0) * 100.0
    return float(np.sqrt((dd**2).mean()))


def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def time_to_target(equity: pd.Series, target: float, periods_per_year: float = 365.0) -> Dict[str, float]:
    """When (if ever) the curve first closed at or above ``target``."""
    hit = equity[equity >= target]
    if hit.empty:
        return {"target_hit": 0.0, "bars_to_target": float("nan"), "years_to_target": float("nan")}
    first = hit.index[0]
    bars = float(equity.index.get_loc(first))
    return {
        "target_hit": 1.0,
        "bars_to_target": bars,
        "years_to_target": bars / periods_per_year,
        "date_to_target": first,
    }


def trade_stats(trades: pd.DataFrame) -> Dict[str, float]:
    if trades is None or trades.empty:
        return {"n_trades": 0.0, "total_costs": 0.0}
    return {
        "n_trades": float(len(trades)),
        "total_costs": float(trades["cost"].sum()),
        "avg_trade_notional": float(trades["notional"].mean()),
        "stop_exits": float((trades["reason"] == "stop").sum()),
    }


def summarize(result, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
    """One flat dict of headline statistics for a :class:`BacktestResult`."""
    eq, rets = result.equity, result.returns
    ppy = result.exec_config.periods_per_year
    gross = result.weights.abs().sum(axis=1)

    out: Dict[str, float] = {
        "start": eq.index[0],
        "end": eq.index[-1],
        "bars": float(len(eq)),
        "initial_equity": float(result.exec_config.initial_capital),
        "final_equity": float(eq.iloc[-1]),
        "total_return": total_return(eq),
        "cagr": cagr(eq, ppy),
        "ann_vol": ann_volatility(rets, ppy),
        "sharpe": sharpe(rets, ppy),
        "sortino": sortino(rets, ppy),
        "max_drawdown": max_drawdown(eq),
        "calmar": calmar(eq, ppy),
        "ulcer_index": ulcer_index(eq),
        "profit_factor": profit_factor(rets),
        "win_rate_bars": float((rets > 0).mean()),
        "avg_gross_exposure": float(gross.mean()),
        "max_gross_exposure": float(gross.max()),
        "time_in_market": float((gross > 1e-9).mean()),
        "turnover_per_year": float(
            result.trades["notional"].sum() / max(eq.mean(), 1e-9) / (len(eq) / ppy)
        )
        if not result.trades.empty
        else 0.0,
        "cost_drag": float(result.costs.sum().sum() / max(result.exec_config.initial_capital, 1e-9)),
        "bust": float(bool(result.meta.get("bust", False))),
    }
    out.update(trade_stats(result.trades))
    out.update(time_to_target(eq, result.exec_config.target_equity, ppy))

    if benchmark is not None and len(benchmark) == len(eq):
        bench = benchmark.reindex(eq.index).ffill()
        out["benchmark_final_equity"] = float(bench.iloc[-1])
        out["benchmark_total_return"] = total_return(bench)
        out["benchmark_max_drawdown"] = max_drawdown(bench)
        out["excess_return"] = out["total_return"] - out["benchmark_total_return"]
    return out


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #
def monte_carlo_paths(
    returns: pd.Series,
    *,
    initial_capital: float = 100.0,
    target: float = 1000.0,
    n_paths: int = 2000,
    horizon: Optional[int] = None,
    block: int = 20,
    seed: int = 0,
    ruin_equity: float = 1.0,
) -> Dict[str, float]:
    """Stationary block bootstrap of the realised bar returns.

    Resampling in blocks keeps short-horizon autocorrelation (streaks, vol
    clustering) intact, so the spread of outcomes is not artificially narrow.
    Answers the question the headline number hides: *out of many futures that
    look like this backtest, how many reach the target, and how many blow up?*
    """
    r = returns.dropna().to_numpy()
    r = r[np.isfinite(r)]
    if r.size < block * 2:
        return {"n_paths": 0.0}
    horizon = int(horizon or r.size)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon / block))

    finals = np.empty(n_paths)
    hit = np.zeros(n_paths, dtype=bool)
    ruined = np.zeros(n_paths, dtype=bool)
    bars_to_hit = np.full(n_paths, np.nan)
    max_dds = np.empty(n_paths)

    starts_hi = r.size - block
    for p in range(n_paths):
        starts = rng.integers(0, max(starts_hi, 1), size=n_blocks)
        path = np.concatenate([r[s : s + block] for s in starts])[:horizon]
        equity = initial_capital * np.cumprod(1.0 + path)
        equity = np.maximum(equity, 0.0)
        peak = np.maximum.accumulate(equity)
        max_dds[p] = float((equity / np.where(peak > 0, peak, 1.0) - 1.0).min())
        ruin_idx = np.argmax(equity <= ruin_equity) if (equity <= ruin_equity).any() else -1
        hit_idx = np.argmax(equity >= target) if (equity >= target).any() else -1
        if hit_idx >= 0 and (ruin_idx < 0 or hit_idx < ruin_idx):
            hit[p] = True
            bars_to_hit[p] = hit_idx
        if ruin_idx >= 0:
            ruined[p] = True
            equity[ruin_idx:] = 0.0
        finals[p] = equity[-1]

    return {
        "n_paths": float(n_paths),
        "p_hit_target": float(hit.mean()),
        "p_ruin": float(ruined.mean()),
        "median_final_equity": float(np.median(finals)),
        "p05_final_equity": float(np.percentile(finals, 5)),
        "p95_final_equity": float(np.percentile(finals, 95)),
        "median_bars_to_target": float(np.nanmedian(bars_to_hit)) if hit.any() else float("nan"),
        "median_max_drawdown": float(np.median(max_dds)),
    }


def deflated_sharpe(
    observed_sharpe: float, n_trials: int, n_obs: int, periods_per_year: float = 365.0
) -> float:
    """How impressive a Sharpe ratio still looks after ``n_trials`` searches.

    Searching hundreds of configurations produces a good-looking Sharpe by
    chance alone. This returns the probability that the true Sharpe is above
    zero once that search is accounted for (Bailey & Lopez de Prado, simplified
    by assuming Gaussian returns).

    ``observed_sharpe`` is annualised, as everything else in this module is;
    it is converted to per-observation units internally, which is the scale the
    statistic is defined on.
    """
    from math import erf, log, sqrt

    if n_trials < 1 or n_obs < 10:
        return float("nan")
    sr = observed_sharpe / sqrt(max(periods_per_year, 1e-9))
    euler = 0.5772156649
    # expected maximum Sharpe from n_trials independent draws of zero-skill noise
    z = sqrt(2.0 * log(max(n_trials, 2)))
    expected_max = z - (log(log(max(n_trials, 3))) + log(4 * np.pi)) / (2 * z) + euler / z
    sr_std = sqrt((1.0 + 0.5 * sr**2) / max(n_obs - 1, 1))
    stat = (sr - expected_max * sr_std) / max(sr_std, 1e-12)
    return float(0.5 * (1.0 + erf(stat / sqrt(2.0))))


def format_summary(stats: Dict[str, float], title: str = "Backtest") -> str:
    """Human-readable block for notebooks and the CLI."""
    def pct(x):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:,.1f}%"

    def num(x, nd=2):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"

    lines = [
        f"== {title} ==",
        f"  period            {stats.get('start')}  ->  {stats.get('end')}  ({int(stats.get('bars', 0)):,} bars)",
        f"  equity            ${num(stats.get('initial_equity'))}  ->  ${num(stats.get('final_equity'))}",
        f"  total return      {pct(stats.get('total_return'))}    CAGR {pct(stats.get('cagr'))}",
        f"  risk              vol {pct(stats.get('ann_vol'))}   maxDD {pct(stats.get('max_drawdown'))}   ulcer {num(stats.get('ulcer_index'))}",
        f"  risk-adjusted     Sharpe {num(stats.get('sharpe'))}   Sortino {num(stats.get('sortino'))}   Calmar {num(stats.get('calmar'))}",
        f"  activity          {int(stats.get('n_trades', 0)):,} trades   turnover {num(stats.get('turnover_per_year'))}x/yr   "
        f"in-market {pct(stats.get('time_in_market'))}   avg gross {num(stats.get('avg_gross_exposure'))}x",
        f"  costs paid        ${num(stats.get('total_costs'))}  ({pct(stats.get('cost_drag'))} of starting capital)",
    ]
    if stats.get("target_hit"):
        lines.append(
            f"  TARGET REACHED    after {int(stats.get('bars_to_target', 0)):,} bars "
            f"({num(stats.get('years_to_target'))} years) on {stats.get('date_to_target')}"
        )
    else:
        lines.append("  target            NOT reached")
    if stats.get("bust"):
        lines.append("  !! ACCOUNT WAS WIPED OUT during this run")
    if "benchmark_final_equity" in stats:
        lines.append(
            f"  buy & hold        ${num(stats['benchmark_final_equity'])}  "
            f"({pct(stats['benchmark_total_return'])}, maxDD {pct(stats.get('benchmark_max_drawdown'))})"
        )
    return "\n".join(lines)
