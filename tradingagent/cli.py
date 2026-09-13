"""Command-line entry point.

    python -m tradingagent.cli --symbols BTC-USD --capital 100 --target 1000
    python -m tradingagent.cli --symbols BTC-USD ETH-USD --mode walkforward --candidates 200
    python -m tradingagent.cli --source synthetic --mode single   # works offline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from .agent import AgentConfig, PortfolioAgent, TradingAgent
from .data import align_universe, bars_per_year, load_universe
from .engine import ExecutionConfig, buy_and_hold_equity
from .metrics import deflated_sharpe, format_summary, monte_carlo_paths
from .optimize import WalkForwardConfig, walk_forward
from .execution import cost_scenario
from .robustness import classify_regimes, era_report, regime_report, robustness_battery
from .universe import UNIVERSES, load_panel
from .xs_optimize import (
    XS_SEARCH_SPACE,
    XS_SEARCH_SPACE_LONG_ONLY,
    XSWalkForwardConfig,
    equal_weight_benchmark,
    walk_forward_xs,
)
from .risk import RiskConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tradingagent",
        description="Backtest a compounding trading agent from a small starting stake.",
    )
    data = p.add_argument_group("data")
    data.add_argument("--symbols", nargs="+", default=["BTC-USD"], help="one or more symbols")
    data.add_argument("--source", default="coinbase", choices=["coinbase", "yahoo", "csv", "synthetic"])
    data.add_argument("--interval", default="1d", help="bar size, e.g. 1d / 6h / 1h")
    data.add_argument("--start", default="2015-01-01")
    data.add_argument("--end", default=None)
    data.add_argument("--refresh", action="store_true", help="ignore the on-disk cache")

    acct = p.add_argument_group("account")
    acct.add_argument("--capital", type=float, default=100.0, help="starting equity in USD")
    acct.add_argument("--target", type=float, default=1000.0, help="profit goal in USD")
    acct.add_argument("--fee-bps", type=float, default=10.0, help="exchange fee per side")
    acct.add_argument("--slippage-bps", type=float, default=5.0)
    acct.add_argument("--max-leverage", type=float, default=2.0)
    acct.add_argument("--stop-at-target", action="store_true", help="stop trading once the goal is hit")
    acct.add_argument("--cost-scenario", default="base", choices=["low", "base", "high"],
                      help="cost model to charge; 'base' reproduces the historical assumption")
    acct.add_argument("--fill-at", default="next_open", choices=["next_open", "next_close"],
                      help="where a scheduled order executes relative to the signal bar")

    run = p.add_argument_group("run")
    run.add_argument(
        "--mode", default="walkforward",
        choices=["walkforward", "single", "cross-section", "robustness"],
        help="cross-section ranks a universe against itself; the others trade each symbol on its own",
    )
    run.add_argument("--universe", default=None,
                     help="named universe for cross-section mode: us_large_cap, crypto_majors, ...")
    run.add_argument("--allow-short", action="store_true",
                     help="let the cross-sectional search use short positions (needs a margin account)")
    run.add_argument("--candidates", type=int, default=150, help="configurations searched per fold")
    run.add_argument("--top-k", type=int, default=5, help="best configurations blended per fold")
    run.add_argument("--train-bars", type=int, default=730)
    run.add_argument("--test-bars", type=int, default=182)
    run.add_argument("--objective", default="target_growth",
                     choices=["target_growth", "growth", "sharpe", "calmar"])
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--quiet", action="store_true")

    out = p.add_argument_group("output")
    out.add_argument("--outdir", default="results")
    out.add_argument("--no-plots", action="store_true")
    out.add_argument("--robust-starts", nargs="*", default=None,
                     help="extra start dates for the robustness battery")
    out.add_argument("--robust-ends", nargs="*", default=None,
                     help="extra end dates for the robustness battery")
    return p


def _load(args) -> Dict[str, pd.DataFrame]:
    kwargs = {"refresh": args.refresh}
    if args.source == "synthetic":
        kwargs = {"n": 2500}
    universe = load_universe(
        args.symbols, interval=args.interval, start=args.start, end=args.end,
        source=args.source, **kwargs,
    )
    return align_universe(universe) if len(universe) > 1 else universe


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ppy = bars_per_year(args.interval)

    if args.mode == "cross-section":
        exec_cfg = ExecutionConfig(
            initial_capital=args.capital,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            max_leverage=args.max_leverage,
            periods_per_year=ppy,
            target_equity=args.target,
            stop_at_target=args.stop_at_target,
        )
    if args.mode == "cross-section":
        names = args.universe or args.symbols
        panel = load_panel(
            names, interval=args.interval, start=args.start, end=args.end,
            source=args.source, min_bars=max(300, args.train_bars // 2),
        )
        if len(panel.symbols) < 12:
            print(
                f"\n  WARNING: only {len(panel.symbols)} symbols loaded. Ranking needs breadth - "
                "below ~20 names\n  the 'cross-section' is a handful of positions and the result is "
                "mostly noise.\n"
            )
        xs = walk_forward_xs(
            panel,
            exec_cfg,
            XSWalkForwardConfig(
                train_bars=args.train_bars, test_bars=args.test_bars,
                n_candidates=args.candidates, top_k=args.top_k,
                objective=args.objective, seed=args.seed, verbose=not args.quiet,
            ),
            XS_SEARCH_SPACE if args.allow_short else XS_SEARCH_SPACE_LONG_ONLY,
        )
        equity, returns, weights, folds = xs.equity, xs.returns, xs.weights, xs.folds
        bench = equal_weight_benchmark(panel, args.capital).reindex(equity.index)
        bench = bench / bench.iloc[0] * args.capital
        stats = xs.stats(benchmark=bench)
        title = f"{len(panel.symbols)} names - cross-sectional walk-forward (out of sample)"
        n_trials = xs.n_evaluations
        n_distinct = int(xs.meta.get("n_candidates", args.candidates))
        print()
        print(format_summary(stats, title))
        print("\n  the benchmark above is an equal-weight basket of the same universe -")
        print("  beating that, not beating one stock, is what says the ranking added something.")
        stability = xs.coefficient_stability()
        print(
            f"\n  model stability across folds: consecutive coefficient correlation "
            f"{stability.get('mean_consecutive_correlation', float('nan')):.2f}"
        )
        print("  models chosen per fold: " + str(folds["model"].value_counts().to_dict()))
        _finish(args, equity, returns, weights, folds, stats, title, n_trials, n_distinct, ppy, bench)
        return 0


    universe = _load(args)
    symbols = list(universe)
    data = universe if len(symbols) > 1 else universe[symbols[0]]
    first = universe[symbols[0]]
    print(f"[data] {', '.join(symbols)}  {len(first):,} bars  "
          f"{first.index[0].date()} -> {first.index[-1].date()}")

    exec_cfg = ExecutionConfig(
        initial_capital=args.capital,
        costs=cost_scenario(args.cost_scenario),
        max_leverage=args.max_leverage,
        periods_per_year=ppy,
        target_equity=args.target,
        stop_at_target=args.stop_at_target,
        fill_at=args.fill_at,
    )

    bench = buy_and_hold_equity(first, args.capital, costs=cost_scenario(args.cost_scenario))

    if args.mode == "robustness":
        return _robustness(args, data, first, exec_cfg, ppy)

    if args.mode == "single":
        risk = RiskConfig(max_leverage=args.max_leverage)
        agent_cfg = AgentConfig(periods_per_year=ppy)
        if len(symbols) > 1:
            result = PortfolioAgent(symbols, agent_cfg, risk).backtest(data, exec_cfg)
        else:
            result = TradingAgent(agent_cfg, risk).backtest(data, exec_cfg)
        equity, returns, weights, folds = result.equity, result.returns, result.weights, None
        stats = result.stats(benchmark=bench)
        title = f"{'+'.join(symbols)} - default agent (in-sample)"
        n_trials = n_distinct = 1
    else:
        wf = walk_forward(
            data, exec_cfg,
            WalkForwardConfig(
                train_bars=args.train_bars, test_bars=args.test_bars,
                n_candidates=args.candidates, top_k=args.top_k,
                objective=args.objective, seed=args.seed, verbose=not args.quiet,
            ),
        )
        equity, returns, weights, folds = wf.equity, wf.returns, wf.weights, wf.folds
        stats = wf.stats(benchmark=bench.reindex(wf.equity.index))
        title = f"{'+'.join(symbols)} - walk-forward (out of sample)"
        n_trials = wf.n_evaluations
        n_distinct = int(wf.meta.get("n_candidates", args.candidates))

    print()
    print(format_summary(stats, title))
    _finish(args, equity, returns, weights, folds, stats, title, n_trials, n_distinct, ppy, bench)
    return 0


def _robustness(args, data, first, exec_cfg, ppy) -> int:
    """Run the full robustness battery plus a regime decomposition."""
    from dataclasses import replace as _replace

    from .agent import AgentConfig, TradingAgent
    from .risk import RiskConfig

    print("\n=== robustness battery ===")
    print("  Not a leaderboard. The useful reading is WHICH scenarios break it.\n")

    def run(**kwargs):
        costs = kwargs.pop("costs", None)
        fill_at = kwargs.pop("fill_at", exec_cfg.fill_at)
        start, end = kwargs.pop("start", None), kwargs.pop("end", None)
        kwargs.pop("seed", None)
        frame = data if isinstance(data, pd.DataFrame) else first
        window = frame.loc[start:end]
        cfg = _replace(exec_cfg, costs=costs or exec_cfg.cost_model(), fill_at=fill_at)
        agent = TradingAgent(AgentConfig(periods_per_year=ppy), RiskConfig(max_leverage=args.max_leverage))
        return agent.backtest(window, cfg).stats()

    battery = robustness_battery(
        run,
        seeds=(),
        start_dates=(None, *(args.robust_starts or [])),
        end_dates=(None, *(args.robust_ends or [])),
        verbose=True,
    )

    print("\n=== regimes ===")
    print("  Where it works, where it fails, and whether the failures are the")
    print("  ones the strategy's design predicts.\n")
    agent = TradingAgent(AgentConfig(periods_per_year=ppy), RiskConfig(max_leverage=args.max_leverage))
    result = agent.backtest(first, exec_cfg)
    regimes = classify_regimes(first["close"], periods_per_year=ppy)
    report = regime_report(result.equity, regimes, periods_per_year=ppy)
    if not report.empty:
        print(report.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    eras = era_report(result.equity, periods_per_year=ppy)
    if not eras.empty:
        print()
        print(eras.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    os.makedirs(args.outdir, exist_ok=True)
    battery.to_csv(os.path.join(args.outdir, "robustness.csv"), index=False)
    report.to_csv(os.path.join(args.outdir, "regimes.csv"), index=False)
    print(f"\n  saved {os.path.join(args.outdir, 'robustness.csv')} and regimes.csv")
    return 0


def _finish(args, equity, returns, weights, folds, stats, title, n_trials, n_distinct, ppy, bench=None):
    """Shared reporting tail: bootstrap, deflation, files and plots."""
    mc = monte_carlo_paths(
        returns, initial_capital=args.capital, target=args.target, n_paths=2000, seed=args.seed
    )
    if mc.get("n_paths"):
        print(
            f"\n  bootstrap ({int(mc['n_paths']):,} resampled histories of the same edge)\n"
            f"    reached ${args.target:,.0f}   {mc['p_hit_target']:.0%}\n"
            f"    wiped out          {mc['p_ruin']:.0%}\n"
            f"    final equity       5th ${mc['p05_final_equity']:,.0f}  "
            f"median ${mc['median_final_equity']:,.0f}  95th ${mc['p95_final_equity']:,.0f}"
        )
    if n_trials > 1:
        # Two defensible ways to count "how many things did we try": the number
        # of distinct configurations (optimistic - the same ones are re-scored
        # each fold) and the number of individual evaluations (pessimistic -
        # each fold makes its own selection). The truth is between them, so
        # report the interval rather than pick the flattering end.
        n_obs = int(stats.get("bars", 0))
        sharpe_val = stats.get("sharpe", 0.0)
        hi = deflated_sharpe(sharpe_val, n_distinct, n_obs, ppy)
        lo = deflated_sharpe(sharpe_val, n_trials, n_obs, ppy)
        print(
            f"    deflated Sharpe: {min(lo, hi):.2f} - {max(lo, hi):.2f} "
            f"(counting {n_distinct:,} distinct configurations to {n_trials:,} evaluations)"
        )
        if max(lo, hi) < 0.5:
            print("    -> below 0.5 at both ends: this Sharpe is within what the search alone could produce.")

    os.makedirs(args.outdir, exist_ok=True)
    equity.to_csv(os.path.join(args.outdir, "equity.csv"))
    if folds is not None:
        folds.to_csv(os.path.join(args.outdir, "folds.csv"), index=False)
    with open(os.path.join(args.outdir, "stats.json"), "w") as fh:
        json.dump({k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in stats.items()}, fh, indent=2)

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        from .report import tearsheet

        path = os.path.join(args.outdir, "tearsheet.png")
        tearsheet(
            equity, benchmark=bench, weights=weights, folds=folds, returns=returns,
            target=args.target, initial=args.capital, title=title, save_path=path,
        )
        print(f"\n  saved {path}")
    print(f"  saved {os.path.join(args.outdir, 'equity.csv')} and stats.json")


if __name__ == "__main__":
    sys.exit(main())
