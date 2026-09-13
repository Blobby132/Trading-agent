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


def trade_stats(trades: pd.DataFrame, meta: Optional[Dict] = None) -> Dict[str, float]:
    if trades is None or trades.empty:
        # a run with record_trades=False keeps only the counters
        meta = meta or {}
        return {
            "n_trades": float(meta.get("n_trades", 0.0)),
            "total_costs": float(meta.get("total_costs", 0.0)),
        }
    return {
        "n_trades": float(len(trades)),
        "total_costs": float(trades["cost"].sum()),
        "avg_trade_notional": float(trades["notional"].mean()),
        "stop_exits": float((trades["reason"] == "stop").sum()),
    }


def _traded_notional(result) -> float:
    """Total notional traded, from the log when it exists and the counter otherwise."""
    if result.trades is not None and not result.trades.empty:
        return float(result.trades["notional"].sum())
    return float((result.meta or {}).get("traded_notional", 0.0))


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
            _traded_notional(result) / max(eq.mean(), 1e-9) / max(len(eq) / ppy, 1e-9)
        ),
        "cost_drag": float(result.costs.sum().sum() / max(result.exec_config.initial_capital, 1e-9)),
        "bust": float(bool(result.meta.get("bust", False))),
    }
    out.update(trade_stats(result.trades, result.meta))
    if not result.trades.empty:
        out["total_costs"] = float(result.trades["cost"].sum())
    elif "fees" in getattr(result.costs, "columns", []):
        out["total_costs"] = float(result.costs.sum().sum())
    out.update(time_to_target(eq, result.exec_config.target_equity, ppy))

    if benchmark is not None and len(benchmark) == len(eq):
        # Rebase the benchmark to the same starting capital on the same date.
        # Slicing a buy-and-hold curve that began years earlier and comparing
        # its level to an account that starts here is not a comparison - it
        # silently credits the benchmark with everything it made before the
        # strategy was even trading.
        bench = benchmark.reindex(eq.index).ffill()
        if len(bench) and bench.iloc[0] not in (0, np.nan) and np.isfinite(bench.iloc[0]):
            bench = bench / bench.iloc[0] * eq.iloc[0]
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


# --------------------------------------------------------------------------- #
# drawdown shape
# --------------------------------------------------------------------------- #
def drawdown_stats(equity: pd.Series, periods_per_year: float = 252.0) -> Dict[str, float]:
    """How deep, how long, and how long to get back.

    Depth is what gets quoted; **duration is what gets people to switch the
    system off.** A 20% drawdown that recovers in a month is a different
    experience from a 20% drawdown that lasts three years, and only one of them
    survives contact with a human.
    """
    if len(equity) < 2:
        return {"max_drawdown": float("nan")}
    peak = equity.cummax()
    dd = equity / peak - 1.0
    underwater = dd < -1e-12

    spells, current = [], 0
    for flag in underwater:
        if flag:
            current += 1
        elif current:
            spells.append(current)
            current = 0
    if current:
        spells.append(current)         # still underwater at the end

    trough = int(np.argmin(dd.to_numpy()))
    recovered = equity.iloc[trough:] >= peak.iloc[trough]
    time_to_recover = (
        float(np.argmax(recovered.to_numpy())) if recovered.any() else float("nan")
    )
    return {
        "max_drawdown": float(dd.min()),
        "max_drawdown_bars": float(max(spells)) if spells else 0.0,
        "max_drawdown_years": float(max(spells) / periods_per_year) if spells else 0.0,
        "avg_drawdown_bars": float(np.mean(spells)) if spells else 0.0,
        "time_underwater": float(underwater.mean()),
        "bars_to_recover_worst": time_to_recover,
        "still_underwater": bool(underwater.iloc[-1]),
    }


def downside_volatility(returns: pd.Series, periods_per_year: float) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    return float(downside.std(ddof=0) * np.sqrt(periods_per_year))


# --------------------------------------------------------------------------- #
# trade-level statistics
# --------------------------------------------------------------------------- #
def round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct completed round trips from the fill log.

    Bar-level win rate - which is what ``summarize`` reported - answers "how
    often was today green", which is not the question a trader asks. This pairs
    fills per symbol into positions opened and closed, so expectancy and the
    win/loss distribution can be computed on the thing that actually has a
    profit and loss.
    """
    if trades is None or trades.empty:
        return pd.DataFrame(
            columns=["symbol", "opened", "closed", "units", "entry", "exit", "pnl", "return", "bars"]
        )

    frame = trades.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in frame.columns:
        frame["timestamp"] = trades.index
    rows = []
    for symbol, group in frame.groupby("symbol", sort=False):
        position = 0.0
        basis = 0.0          # signed cost of the open position
        opened_at = None
        for _, fill in group.sort_values("timestamp").iterrows():
            units, price = float(fill["units"]), float(fill["price"])
            if position == 0.0:
                position, basis, opened_at = units, units * price, fill["timestamp"]
                continue
            if np.sign(units) == np.sign(position):          # adding
                position += units
                basis += units * price
                continue
            # reducing or closing
            closing = min(abs(units), abs(position)) * np.sign(position)
            entry_price = basis / position if position else price
            pnl = closing * (price - entry_price)
            rows.append({
                "symbol": symbol, "opened": opened_at, "closed": fill["timestamp"],
                "units": abs(closing), "entry": entry_price, "exit": price, "pnl": float(pnl),
                "return": float((price / entry_price - 1.0) * np.sign(position))
                if entry_price else 0.0,
            })
            basis -= closing * entry_price
            position -= closing
            remainder = units + closing
            if abs(position) < 1e-12 and abs(remainder) > 1e-12:   # flipped through zero
                position, basis, opened_at = remainder, remainder * price, fill["timestamp"]
            elif abs(position) < 1e-12:
                position, basis, opened_at = 0.0, 0.0, None

    result = pd.DataFrame(rows)
    if not result.empty:
        result["bars"] = (
            pd.to_datetime(result["closed"]) - pd.to_datetime(result["opened"])
        ).dt.days
    return result


def trade_level_stats(trades: pd.DataFrame) -> Dict[str, float]:
    """Expectancy and the win/loss distribution, per completed round trip."""
    trips = round_trips(trades)
    if trips.empty:
        return {"round_trips": 0.0}
    wins = trips[trips["pnl"] > 0]["pnl"]
    losses = trips[trips["pnl"] < 0]["pnl"]
    win_rate = float(len(wins) / len(trips))
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    return {
        "round_trips": float(len(trips)),
        "win_rate_trades": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": float(wins.max()) if len(wins) else 0.0,
        "largest_loss": float(losses.min()) if len(losses) else 0.0,
        # what one trade is worth on average, in currency
        "expectancy": float(win_rate * avg_win + (1.0 - win_rate) * avg_loss),
        "payoff_ratio": float(abs(avg_win / avg_loss)) if avg_loss else float("inf"),
        "avg_holding_bars": float(trips["bars"].mean()) if "bars" in trips else float("nan"),
    }


# --------------------------------------------------------------------------- #
# uncertainty
# --------------------------------------------------------------------------- #
def bootstrap_stats(
    returns: pd.Series,
    *,
    periods_per_year: float = 252.0,
    n_boot: int = 2000,
    block: int = 20,
    seed: int = 0,
    confidence: float = 0.90,
) -> Dict[str, float]:
    """Confidence intervals for the headline statistics.

    A Sharpe ratio is an estimate, and over a few hundred bars it is a noisy
    one. Quoting it without an interval invites the reader to treat a number
    that could plausibly be 0.2 or 1.6 as though it were 0.9.

    Resampling is done in blocks, which keeps volatility clustering and streaks
    intact; an i.i.d. bootstrap would report intervals that are far too tight.
    """
    series = returns.dropna()
    series = series[np.isfinite(series)]
    if len(series) < block * 3:
        return {"n_boot": 0.0}

    values = series.to_numpy()
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(values) / block))
    sharpes = np.empty(n_boot)
    cagrs = np.empty(n_boot)
    max_dds = np.empty(n_boot)

    for i in range(n_boot):
        starts = rng.integers(0, max(len(values) - block, 1), size=n_blocks)
        path = np.concatenate([values[s : s + block] for s in starts])[: len(values)]
        sd = path.std(ddof=0)
        sharpes[i] = (path.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
        equity = np.cumprod(1.0 + path)
        years = len(path) / periods_per_year
        cagrs[i] = equity[-1] ** (1.0 / years) - 1.0 if equity[-1] > 0 and years > 0 else -1.0
        peak = np.maximum.accumulate(equity)
        max_dds[i] = (equity / peak - 1.0).min()

    lo_q, hi_q = (1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100
    return {
        "n_boot": float(n_boot),
        "confidence": float(confidence),
        "sharpe_lo": float(np.percentile(sharpes, lo_q)),
        "sharpe_hi": float(np.percentile(sharpes, hi_q)),
        "sharpe_p_positive": float((sharpes > 0).mean()),
        "cagr_lo": float(np.percentile(cagrs, lo_q)),
        "cagr_hi": float(np.percentile(cagrs, hi_q)),
        "max_drawdown_p05": float(np.percentile(max_dds, 5)),
        "max_drawdown_median": float(np.median(max_dds)),
    }


def full_report(result, benchmark: Optional[pd.Series] = None, *, bootstrap: bool = True) -> Dict[str, float]:
    """Everything :func:`summarize` reports, plus shape, trades and uncertainty."""
    stats = summarize(result, benchmark=benchmark)
    ppy = result.exec_config.periods_per_year
    stats.update(drawdown_stats(result.equity, ppy))
    stats["downside_vol"] = downside_volatility(result.returns, ppy)
    stats.update(trade_level_stats(result.trades))

    costs = getattr(result, "costs", None)
    if costs is not None and not costs.empty:
        for column in costs.columns:
            stats[f"cost_{column}"] = float(costs[column].sum())
    gross = stats.get("total_return", float("nan"))
    paid = stats.get("total_costs", 0.0) + stats.get("cost_financing", 0.0)
    initial = max(stats.get("initial_equity", 1.0), 1e-9)
    stats["net_return"] = gross
    stats["gross_return"] = gross + paid / initial
    weights = result.weights
    stats["avg_position_size"] = (
        float(weights.abs().replace(0.0, np.nan).stack().mean()) if not weights.empty else 0.0
    )
    if bootstrap:
        stats.update(bootstrap_stats(result.returns, periods_per_year=ppy))
    return stats


def format_full_report(stats: Dict[str, float], title: str = "Backtest") -> str:
    """The long form: performance, shape, trades, costs and uncertainty."""
    def pct(key, nd=1):
        value = stats.get(key)
        return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value * 100:,.{nd}f}%"

    def num(key, nd=2):
        value = stats.get(key)
        return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:,.{nd}f}"

    lines = [
        format_summary(stats, title),
        "",
        "  -- drawdown shape ------------------------------------------------",
        f"    longest drawdown  {num('max_drawdown_bars', 0)} bars "
        f"({num('max_drawdown_years')} years)   average {num('avg_drawdown_bars', 0)} bars",
        f"    time underwater   {pct('time_underwater')}"
        + ("   (still underwater at the end)" if stats.get("still_underwater") else ""),
        f"    downside vol      {pct('downside_vol')}",
        "",
        "  -- trades --------------------------------------------------------",
        f"    round trips       {num('round_trips', 0)}   win rate {pct('win_rate_trades')}",
        f"    average win       {num('avg_win')}      average loss {num('avg_loss')}",
        f"    largest win       {num('largest_win')}      largest loss {num('largest_loss')}",
        f"    expectancy        {num('expectancy')} per trade   payoff {num('payoff_ratio')}",
        f"    avg holding       {num('avg_holding_bars', 0)} days   avg position {pct('avg_position_size')}",
        "",
        "  -- costs ---------------------------------------------------------",
        f"    gross return      {pct('gross_return')}   net {pct('net_return')}",
        f"    fees {num('cost_fees')}   financing {num('cost_financing')}   "
        f"total {num('total_costs')}",
    ]
    if stats.get("n_boot"):
        lines += [
            "",
            "  -- uncertainty (block bootstrap) ---------------------------------",
            f"    Sharpe            {num('sharpe')}  "
            f"[{num('sharpe_lo')}, {num('sharpe_hi')}] at {pct('confidence', 0)} confidence",
            f"    P(Sharpe > 0)     {pct('sharpe_p_positive')}",
            f"    CAGR              {pct('cagr')}  [{pct('cagr_lo')}, {pct('cagr_hi')}]",
            f"    plausible maxDD   median {pct('max_drawdown_median')}, "
            f"5th pct {pct('max_drawdown_p05')}",
        ]
        if stats.get("sharpe_lo", 0) <= 0:
            lines.append(
                "    NOTE: the interval includes zero - this sample does not "
                "establish an edge."
            )
    return "\n".join(lines)
