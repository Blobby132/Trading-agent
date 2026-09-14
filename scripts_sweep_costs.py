"""Does hourly's gross edge survive a better cost structure?

The grid's finding is that hourly extracts more signal than daily and cannot pay
for it at 15 bps a side. That is a statement about a retail taker, not about the
market, so it has an obvious test: re-run the contenders under the LOW scenario
(4 bps a side - a maker-priced fill) and see whether hourly's gross edge
survives contact with it.

If it does, the finding becomes "hourly needs a better cost structure", which is
actionable. If it does not, hourly is finished as a direction.
"""
import json, os
from concurrent.futures import ProcessPoolExecutor, as_completed

from tradingagent.data import load_prices
from tradingagent.engine import ExecutionConfig
from tradingagent.execution import HIGH_COST, LOW_COST
from tradingagent.optimize import WalkForwardConfig
from tradingagent.sweep import Cell, clip, common_range, run_cell

SCENARIOS = {"low": LOW_COST, "high": HIGH_COST}
CELLS = [("1h", 6), ("1d", 1)]
SEEDS = [0, 1]
N_CAND = 64
OUT = "results/sweeps/BTC-USD_costs.jsonl"


def _one(args):
    interval, every, scenario, seed, start, end = args
    frame = clip(load_prices("BTC-USD", interval=interval, start="2015-01-01", source="coinbase"), start, end)
    res = run_cell(
        frame, Cell(interval, every),
        base_exec=ExecutionConfig(initial_capital=100.0, costs=SCENARIOS[scenario]),
        wf_daily=WalkForwardConfig(n_candidates=N_CAND, seed=seed, verbose=False),
    )
    row = res.row()
    row.update(seed=seed, cost_scenario=scenario, bars=float(res.stats.get("bars", 0.0)))
    return row


if __name__ == "__main__":
    frames = {iv: load_prices("BTC-USD", interval=iv, start="2015-01-01", source="coinbase")
              for iv in ("1d", "6h", "1h")}
    start, end = common_range(frames)
    del frames
    jobs = [(iv, ev, sc, sd, start, end) for iv, ev in CELLS for sc in SCENARIOS for sd in SEEDS]
    jobs.sort(key=lambda j: 0 if j[0] == "1h" else 1)
    with open(OUT, "w") as fh, ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_one, j): j for j in jobs}
        for fut in as_completed(futures):
            iv, ev, sc, sd, _, _ = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                print(f"[costs] FAILED {iv}/{ev}/{sc}/seed{sd}: {exc}", flush=True); continue
            fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
            print(f"[costs] {iv}/every{ev} {sc:>4} seed {sd}: net ${row['final_equity']:,.0f} "
                  f"gross ${row['gross_final_equity']:,.0f} Sharpe {row['sharpe']:.2f} "
                  f"keeps {row['final_equity']/max(row['gross_final_equity'],1e-9):.0%} of gross",
                  flush=True)
    print("COSTS DONE", flush=True)
