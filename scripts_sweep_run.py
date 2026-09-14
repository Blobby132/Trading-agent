"""Run the interval x frequency grid for the crypto agent.

Single seed, one constant candidate budget per cell, one common date range.
Written to results/sweeps/ as it goes so a long run can be inspected mid-flight.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from tradingagent.data import load_prices
from tradingagent.optimize import WalkForwardConfig
from tradingagent.sweep import Cell, clip, common_range, coverage_table, run_cell

SYMBOL = os.environ.get("SWEEP_SYMBOL", "BTC-USD")
N_CAND = int(os.environ.get("SWEEP_CANDIDATES", "64"))
SEED = int(os.environ.get("SWEEP_SEED", "0"))
OUT = os.environ.get("SWEEP_OUT", f"results/sweeps/{SYMBOL}_seed{SEED}.jsonl")

GRID = {"1d": [1, 5], "6h": [1, 2, 4, 20], "1h": [1, 2, 3, 6, 12, 24, 120]}


def _one(args):
    interval, every, start, end = args
    frame = clip(load_prices(SYMBOL, interval=interval, start="2015-01-01", source="coinbase"), start, end)
    cell = Cell(interval, every)
    res = run_cell(frame, cell, wf_daily=WalkForwardConfig(n_candidates=N_CAND, seed=SEED, verbose=False))
    row = res.row()
    row["seed"] = SEED
    row["symbol"] = SYMBOL
    row["bars"] = float(res.stats.get("bars", 0.0))
    row["sortino"] = res.stats.get("sortino")
    row["time_in_market"] = res.stats.get("time_in_market")
    row["avg_gross_exposure"] = res.stats.get("avg_gross_exposure")
    return row


if __name__ == "__main__":
    frames = {iv: load_prices(SYMBOL, interval=iv, start="2015-01-01", source="coinbase") for iv in GRID}
    start, end = common_range(frames)
    print(f"[grid] {SYMBOL} common range {start} -> {end}", flush=True)
    print(coverage_table(frames).to_string(), flush=True)
    del frames

    jobs = [(iv, every, start, end) for iv, everys in GRID.items() for every in everys]
    # heaviest first so the tail of the run is short jobs
    jobs.sort(key=lambda j: {"1h": 0, "6h": 1, "1d": 2}[j[0]])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = 0
    with open(OUT, "w") as fh, ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_one, j): j for j in jobs}
        for fut in as_completed(futures):
            iv, every, _, _ = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                print(f"[grid] FAILED {iv}/every{every}: {type(exc).__name__}: {exc}", flush=True)
                continue
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            done += 1
            print(
                f"[grid {done}/{len(jobs)}] {iv}/every{every}: net ${row['final_equity']:,.0f} "
                f"gross ${row['gross_final_equity']:,.0f} Sharpe {row['sharpe']:.2f} "
                f"DD {row['max_drawdown']:.1%} trades/day {row['actual_trades_day']:.2f} "
                f"cost {row['cost_drag_pct_capital']:.0%} ({row['seconds']:.0f}s)",
                flush=True,
            )
    print("GRID DONE", flush=True)
