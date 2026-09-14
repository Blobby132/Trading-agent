"""Rebalance-frequency sweep for the cross-sectional stock agent.

Daily bars only - see the report for why an intraday version is not runnable
from this environment. ``rebalance_every`` is PINNED per cell rather than left
in the search space, so the frequency is the experiment instead of something
the optimiser quietly opts out of.
"""
import json, os, time

import pandas as pd

from tradingagent.engine import ExecutionConfig
from tradingagent.universe import load_panel
from tradingagent.xs_optimize import (
    XS_SEARCH_SPACE_LONG_ONLY,
    XSWalkForwardConfig,
    walk_forward_xs,
)

EVERYS = [1, 2, 3, 5, 10, 21, 63]
N_CAND = int(os.environ.get("XS_CANDIDATES", "48"))
SEED = int(os.environ.get("XS_SEED", "0"))
OUT = f"results/sweeps/stocks_seed{SEED}.jsonl"

if __name__ == "__main__":
    panel = load_panel("us_large_cap", source="nasdaq", start="2016-01-01", pause=0.05)
    print(f"[stocks] {len(panel.symbols)} names, {len(panel.index)} bars "
          f"{panel.index[0].date()} -> {panel.index[-1].date()}", flush=True)
    base = ExecutionConfig(initial_capital=100.0, periods_per_year=252.0, record_trades=False)
    os.makedirs("results/sweeps", exist_ok=True)
    with open(OUT, "w") as fh:
        for every in EVERYS:
            space = {**XS_SEARCH_SPACE_LONG_ONLY, "rebalance_every": [every]}
            cfg = XSWalkForwardConfig(n_candidates=N_CAND, seed=SEED, verbose=False)
            t0 = time.time()
            res = walk_forward_xs(panel, base, cfg, space, gross_twin=True)
            s = res.stats()
            gross = res.meta.get("gross_equity")
            gfinal = float(gross.iloc[-1]) if gross is not None else float("nan")
            years = float(s.get("bars", 0)) / 252.0
            costs = float(s.get("total_costs", 0.0)) + float(s.get("financing_paid", 0.0))
            row = {
                "rebalance_every": every,
                "rebalances_per_year": 252.0 / every,
                "final_equity": s.get("final_equity"),
                "gross_final_equity": gfinal,
                "net_return": s.get("total_return"),
                "gross_return": gfinal / 100.0 - 1.0,
                "cagr": s.get("cagr"),
                "sharpe": s.get("sharpe"),
                "max_drawdown": s.get("max_drawdown"),
                "n_trades": s.get("n_trades"),
                "trades_per_day": float(s.get("n_trades", 0.0)) / max(years * 252.0, 1e-9),
                "turnover_per_year": s.get("turnover_per_year"),
                "cost_drag_pct_capital": costs / 100.0,
                "fees_paid": s.get("total_costs"),
                "financing_paid": s.get("financing_paid"),
                "oos_years": years,
                "bars": s.get("bars"),
                "evaluations": res.n_evaluations,
                "target_hit": s.get("target_hit"),
                "seconds": time.time() - t0,
            }
            fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
            print(f"[stocks] every={every:>2}: net ${row['final_equity']:,.0f} "
                  f"gross ${row['gross_final_equity']:,.0f} Sharpe {row['sharpe']:.2f} "
                  f"DD {row['max_drawdown']:.1%} trades/day {row['trades_per_day']:.2f} "
                  f"cost {row['cost_drag_pct_capital']:.0%} ({row['seconds']:.0f}s)", flush=True)
    print("STOCKS DONE", flush=True)
