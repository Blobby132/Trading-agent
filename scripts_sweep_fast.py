"""The high-frequency arm: same grid, dust filter loosened.

The main grid pins ``min_trade_frac`` at 0.10 - the repository default, which
ignores any rebalance smaller than a tenth of the position being adjusted. At
hourly bars most bar-to-bar weight changes are smaller than that, so the filter
silently absorbs the frequency knob: cells asking for 24 rebalances a day
execute about one. Reporting "high frequency did not help" off that grid would
be reporting on an experiment that never ran.

So this arm re-runs the fast cells with the filter at 0.01, which lets the
intended trade rate actually happen, and pays the costs that come with it. It is
the honest version of the question "does trading 10+ times a day help?" - and
the answer costs what it costs.
"""
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tradingagent.data import load_prices
from tradingagent.engine import ExecutionConfig
from tradingagent.optimize import DEFAULT_SEARCH_SPACE, WalkForwardConfig
from tradingagent.sweep import Cell, clip, common_range, run_cell

N_CAND = int(os.environ.get("SWEEP_CANDIDATES", "64"))
SEED = int(os.environ.get("SWEEP_SEED", "0"))
MIN_TRADE = float(os.environ.get("SWEEP_MIN_TRADE", "0.01"))
OUT = f"results/sweeps/BTC-USD_fast_mtf{MIN_TRADE}_seed{SEED}.jsonl"
GRID = {"1h": [1, 2, 6, 24], "6h": [1, 4]}


def _one(args):
    interval, every, start, end = args
    frame = clip(load_prices("BTC-USD", interval=interval, start="2015-01-01", source="coinbase"), start, end)
    space = {**DEFAULT_SEARCH_SPACE, "min_trade_frac": [MIN_TRADE]}
    res = run_cell(
        frame, Cell(interval, every),
        base_exec=ExecutionConfig(initial_capital=100.0, min_trade_frac=MIN_TRADE),
        wf_daily=WalkForwardConfig(n_candidates=N_CAND, seed=SEED, verbose=False),
        space_daily=space,
    )
    row = res.row()
    row.update(seed=SEED, min_trade_frac=MIN_TRADE, symbol="BTC-USD",
               bars=float(res.stats.get("bars", 0.0)))
    return row


if __name__ == "__main__":
    frames = {iv: load_prices("BTC-USD", interval=iv, start="2015-01-01", source="coinbase")
              for iv in list(GRID) + ["1d"]}
    start, end = common_range(frames)
    del frames
    jobs = [(iv, e, start, end) for iv, everys in GRID.items() for e in everys]
    jobs.sort(key=lambda j: {"1h": 0, "6h": 1}[j[0]])
    with open(OUT, "w") as fh, ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_one, j): j for j in jobs}
        for fut in as_completed(futures):
            iv, every, _, _ = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                print(f"[fast] FAILED {iv}/every{every}: {type(exc).__name__}: {exc}", flush=True)
                continue
            fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
            print(f"[fast] {iv}/every{every}: net ${row['final_equity']:,.0f} "
                  f"gross ${row['gross_final_equity']:,.0f} Sharpe {row['sharpe']:.2f} "
                  f"DD {row['max_drawdown']:.1%} trades/day {row['actual_trades_day']:.2f} "
                  f"(intended {row['intended_trades_day']:.2g}) "
                  f"cost {row['cost_drag_pct_capital']:.0%} ({row['seconds']:.0f}s)", flush=True)
    print("FAST DONE", flush=True)
