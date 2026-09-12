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

    run = p.add_argument_group("run")
    run.add_argument("--mode", default="walkforward", choices=["walkforward", "single"])
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
    universe = _load(args)
    symbols = list(universe)
    data = universe if len(symbols) > 1 else universe[symbols[0]]
    first = universe[symbols[0]]
    print(f"[data] {', '.join(symbols)}  {len(first):,} bars  "
          f"{first.index[0].date()} -> {first.index[-1].date()}")

    exec_cfg = ExecutionConfig(
        initial_capital=args.capital,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        max_leverage=args.max_leverage,
        periods_per_year=ppy,
        target_equity=args.target,
        stop_at_target=args.stop_at_target,
    )

    bench = buy_and_hold_equity(first, args.capital, args.fee_bps)

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
        n_trials = 1
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

    print()
    print(format_summary(stats, title))

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
        dsr = deflated_sharpe(stats.get("sharpe", 0.0), n_trials, int(stats.get("bars", 0)), ppy)
        print(f"    deflated Sharpe p-value after {n_trials:,} configurations searched: {dsr:.2f}")

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
