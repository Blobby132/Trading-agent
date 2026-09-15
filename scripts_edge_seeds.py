"""More seeds on the surviving candidates.

Three seeds is not enough here: the baseline's own three-seed range on BTC is
$888 to $1,788, so a median difference of a few percent sits well inside it.
This runs the control and the candidates over a wider seed set and reports the
paired difference seed by seed, which removes the shared market path and is far
more sensitive than comparing two medians.
"""
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from tradingagent.data import load_prices
from tradingagent.engine import ExecutionConfig
from tradingagent.optimize import DEFAULT_SEARCH_SPACE, WalkForwardConfig, walk_forward

CANDIDATES = json.loads(os.environ.get("EDGE_CANDIDATES", '{"V0_baseline": {}}'))
SEEDS = [int(s) for s in os.environ.get("EDGE_SEEDS", "0,1,2,3,4,5,6,7").split(",")]
N_CAND = 64
OUT = os.environ.get("EDGE_OUT", "results/edge/seeds.jsonl")


def _one(args):
    name, seed = args
    btc = load_prices("BTC-USD", interval="1d", start="2015-01-01", source="coinbase")
    space = {**DEFAULT_SEARCH_SPACE, **CANDIDATES[name]}
    base = ExecutionConfig(initial_capital=100.0, periods_per_year=365.0, record_trades=False)
    res = walk_forward(btc, base, WalkForwardConfig(n_candidates=N_CAND, seed=seed, verbose=False), space)
    s = res.stats()
    return {"variant": name, "seed": seed, "final_equity": s["final_equity"],
            "cagr": s["cagr"], "sharpe": s["sharpe"], "sortino": s["sortino"],
            "max_drawdown": s["max_drawdown"], "n_trades": s["n_trades"],
            "avg_exposure": s["avg_gross_exposure"], "total_costs": s.get("total_costs")}


if __name__ == "__main__":
    jobs = [(n, s) for n in CANDIDATES for s in SEEDS]
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
                print(f"[seeds] FAILED {n}/{s}: {e}", flush=True); continue
            rows.append(r); fh.write(json.dumps(r, default=str) + "\n"); fh.flush()
            done += 1
            if done % 6 == 0:
                print(f"[seeds] {done}/{len(jobs)}", flush=True)
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\n== per-variant summary ==")
    print(d.groupby("variant").agg(
        seeds=("seed","count"), median_final=("final_equity","median"),
        mean_final=("final_equity","mean"), min_final=("final_equity","min"),
        max_final=("final_equity","max"), median_sharpe=("sharpe","median"),
        median_sortino=("sortino","median"), median_dd=("max_drawdown","median"),
        median_exposure=("avg_exposure","median")).round(3).to_string())

    # paired comparison against the control, seed by seed
    ctrl = "V0_baseline"
    if ctrl in set(d["variant"]):
        wide = d.pivot(index="seed", columns="variant", values="final_equity")
        sh = d.pivot(index="seed", columns="variant", values="sharpe")
        print(f"\n== paired vs {ctrl}: same seed, same market path ==")
        for v in wide.columns:
            if v == ctrl: continue
            diff = (wide[v] / wide[ctrl] - 1.0).dropna()
            dsh = (sh[v] - sh[ctrl]).dropna()
            wins = int((diff > 0).sum())
            t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan
            print(f"  {v:<28} equity {diff.mean():+7.2%} mean, wins {wins}/{len(diff)} seeds, "
                  f"t={t:+.2f}   Sharpe {dsh.mean():+.3f} mean, wins {int((dsh>0).sum())}/{len(dsh)}")
    print("\nSEEDS DONE", flush=True)
