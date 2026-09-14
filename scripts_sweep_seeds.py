"""Re-run named cells across several seeds.

One seed picks one random subset of the search space, so a single-seed grid
tells you the shape of the answer and nothing about its stability. Any cell that
looks like it beats the daily baseline has to survive this before it is worth
believing; the published daily baseline was itself a five-seed median.

Usage: SWEEP_CELLS="1d:1,6h:4,1h:24" python scripts_sweep_seeds.py
"""
import json, os
from concurrent.futures import ProcessPoolExecutor, as_completed

from tradingagent.data import load_prices
from tradingagent.optimize import WalkForwardConfig
from tradingagent.sweep import Cell, clip, common_range, run_cell

CELLS = [c.split(":") for c in os.environ.get("SWEEP_CELLS", "1d:1").split(",")]
SEEDS = [int(s) for s in os.environ.get("SWEEP_SEEDS", "1,2,3").split(",")]
N_CAND = int(os.environ.get("SWEEP_CANDIDATES", "64"))
OUT = os.environ.get("SWEEP_OUT", "results/sweeps/BTC-USD_seeds.jsonl")


def _one(args):
    interval, every, seed, start, end = args
    frame = clip(load_prices("BTC-USD", interval=interval, start="2015-01-01", source="coinbase"), start, end)
    res = run_cell(frame, Cell(interval, every),
                   wf_daily=WalkForwardConfig(n_candidates=N_CAND, seed=seed, verbose=False))
    row = res.row()
    row.update(seed=seed, bars=float(res.stats.get("bars", 0.0)))
    return row


if __name__ == "__main__":
    intervals = sorted({c[0] for c in CELLS} | {"1d", "6h", "1h"})
    frames = {iv: load_prices("BTC-USD", interval=iv, start="2015-01-01", source="coinbase")
              for iv in intervals}
    start, end = common_range(frames)
    del frames
    jobs = [(iv, int(ev), seed, start, end) for iv, ev in CELLS for seed in SEEDS]
    jobs.sort(key=lambda j: {"1h": 0, "6h": 1, "1d": 2, "15m": -1}.get(j[0], 0))
    with open(OUT, "w") as fh, ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_one, j): j for j in jobs}
        for fut in as_completed(futures):
            iv, ev, seed, _, _ = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                print(f"[seeds] FAILED {iv}/every{ev}/seed{seed}: {exc}", flush=True); continue
            fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
            print(f"[seeds] {iv}/every{ev} seed {seed}: net ${row['final_equity']:,.0f} "
                  f"Sharpe {row['sharpe']:.2f} DD {row['max_drawdown']:.1%}", flush=True)
    print("SEEDS DONE", flush=True)
