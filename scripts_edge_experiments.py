"""Hypothesis-driven exposure-shaping experiments.

Each variant fixes the shaping knobs and runs the SAME walk-forward over the
SAME 2,304-configuration space, so the search does not grow and the optimiser
cannot hunt for the shaping that flatters it. Three seeds each, because a
one-seed difference in this repository has already been shown to be noise.
"""
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from tradingagent.data import load_prices
from tradingagent.engine import ExecutionConfig
from tradingagent.ledger import ResearchLedger
from tradingagent.optimize import DEFAULT_SEARCH_SPACE, WalkForwardConfig, space_size, walk_forward

VARIANTS = {
    "V0_baseline":        {"signal_shape": 1.0, "signal_deadband": 0.00},
    "V1_shape1.5":        {"signal_shape": 1.5, "signal_deadband": 0.00},
    "V2_shape2.0":        {"signal_shape": 2.0, "signal_deadband": 0.00},
    "V3_deadband0.15":    {"signal_shape": 1.0, "signal_deadband": 0.15},
    "V4_deadband0.30":    {"signal_shape": 1.0, "signal_deadband": 0.30},
    "V5_shape1.5_dead.15":{"signal_shape": 1.5, "signal_deadband": 0.15},
}
SEEDS = [0, 1, 2]
N_CAND = 64
OUT = "results/edge/variants.jsonl"


def _one(args):
    name, seed = args
    cfg = VARIANTS[name]
    btc = load_prices("BTC-USD", interval="1d", start="2015-01-01", source="coinbase")
    space = {**DEFAULT_SEARCH_SPACE,
             "signal_shape": [cfg["signal_shape"]],
             "signal_deadband": [cfg["signal_deadband"]]}
    base = ExecutionConfig(initial_capital=100.0, periods_per_year=365.0, record_trades=False)
    t0 = time.time()
    res = walk_forward(btc, base, WalkForwardConfig(n_candidates=N_CAND, seed=seed, verbose=False),
                       space, gross_twin=True)
    s = res.stats()
    gross = res.meta.get("gross_equity")
    return {"variant": name, "seed": seed, **cfg,
            "final_equity": s["final_equity"], "cagr": s["cagr"], "sharpe": s["sharpe"],
            "sortino": s["sortino"], "max_drawdown": s["max_drawdown"],
            "n_trades": s["n_trades"], "turnover": s["turnover_per_year"],
            "avg_exposure": s["avg_gross_exposure"], "time_in_market": s["time_in_market"],
            "total_costs": s.get("total_costs"), "bars": s["bars"],
            "gross_final": float(gross.iloc[-1]) if gross is not None else np.nan,
            "evaluations": res.n_evaluations, "space_size": space_size(space),
            "seconds": time.time() - t0}


if __name__ == "__main__":
    jobs = [(n, s) for n in VARIANTS for s in SEEDS]
    os.makedirs("results/edge", exist_ok=True)
    rows = []
    with open(OUT, "w") as fh, ProcessPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_one, j): j for j in jobs}
        done = 0
        for f in as_completed(futs):
            n, s = futs[f]
            try:
                r = f.result()
            except Exception as e:
                print(f"[edge] FAILED {n} seed {s}: {type(e).__name__}: {e}", flush=True); continue
            rows.append(r); fh.write(json.dumps(r, default=str) + "\n"); fh.flush()
            done += 1
            print(f"[edge {done}/{len(jobs)}] {n} seed {s}: ${r['final_equity']:,.0f} "
                  f"CAGR {r['cagr']:.1%} Sharpe {r['sharpe']:.3f} Sortino {r['sortino']:.3f} "
                  f"DD {r['max_drawdown']:.1%} exp {r['avg_exposure']:.3f} "
                  f"trades {int(r['n_trades'])}", flush=True)
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    agg = d.groupby("variant").agg(
        seeds=("seed","count"),
        median_final=("final_equity","median"), min_final=("final_equity","min"),
        max_final=("final_equity","max"),
        median_cagr=("cagr","median"), median_sharpe=("sharpe","median"),
        median_sortino=("sortino","median"), median_dd=("max_drawdown","median"),
        median_exposure=("avg_exposure","median"), median_trades=("n_trades","median"),
        median_turnover=("turnover","median"), median_costs=("total_costs","median"),
    ).round(3)
    print("\n== variants, median of 3 seeds ==")
    print(agg.to_string())
    agg.to_csv("results/edge/variants_summary.csv")
    print("\nEDGE EXPERIMENTS DONE", flush=True)
