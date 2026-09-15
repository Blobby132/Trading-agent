"""Batch 2: the two hypotheses the diagnostics actually pointed at.

H-REGIME  The ablation showed forcing the regime filter ON improves the frozen
          config on return, Sharpe, drawdown AND turnover simultaneously - even
          though the optimiser switched it off in 58% of folds. Pinning it ON
          SHRINKS the search space rather than growing it.

H-FLOOR   Against a plain 12-month momentum rule, the bars where the rule was
          invested and the agent flat are 21.9% of the sample and the asset
          returned +40.2% annualised across them. A floor on long exposure while
          the 12-month trend is up should recover part of that block without
          touching the downtrend behaviour that the agent is actually good at.
"""
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from tradingagent.data import load_prices
from tradingagent.engine import ExecutionConfig
from tradingagent.optimize import DEFAULT_SEARCH_SPACE, WalkForwardConfig, space_size, walk_forward

VARIANTS = {
    "W0_control":            {},
    "W1_regime_on":          {"regime_filter": [1]},
    "W2_floor0.25":          {"trend_floor": [0.25]},
    "W3_floor0.50":          {"trend_floor": [0.50]},
    "W4_regime_floor0.25":   {"regime_filter": [1], "trend_floor": [0.25]},
    "W5_regime_floor0.25_shape1.5": {"regime_filter": [1], "trend_floor": [0.25],
                                     "signal_shape": [1.5]},
}
SEEDS = [0, 1, 2]
N_CAND = 64
OUT = "results/edge/variants2.jsonl"


def _one(args):
    name, seed = args
    btc = load_prices("BTC-USD", interval="1d", start="2015-01-01", source="coinbase")
    space = {**DEFAULT_SEARCH_SPACE, **VARIANTS[name]}
    base = ExecutionConfig(initial_capital=100.0, periods_per_year=365.0, record_trades=False)
    t0 = time.time()
    res = walk_forward(btc, base, WalkForwardConfig(n_candidates=N_CAND, seed=seed, verbose=False),
                       space, gross_twin=True)
    s = res.stats()
    g = res.meta.get("gross_equity")
    return {"variant": name, "seed": seed,
            "final_equity": s["final_equity"], "cagr": s["cagr"], "sharpe": s["sharpe"],
            "sortino": s["sortino"], "max_drawdown": s["max_drawdown"],
            "n_trades": s["n_trades"], "turnover": s["turnover_per_year"],
            "avg_exposure": s["avg_gross_exposure"], "time_in_market": s["time_in_market"],
            "total_costs": s.get("total_costs"), "bars": s["bars"],
            "gross_final": float(g.iloc[-1]) if g is not None else np.nan,
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
                print(f"[edge2] FAILED {n} seed {s}: {type(e).__name__}: {e}", flush=True); continue
            rows.append(r); fh.write(json.dumps(r, default=str) + "\n"); fh.flush()
            done += 1
            print(f"[edge2 {done}/{len(jobs)}] {n} seed {s}: ${r['final_equity']:,.0f} "
                  f"CAGR {r['cagr']:.1%} Sharpe {r['sharpe']:.3f} Sortino {r['sortino']:.3f} "
                  f"DD {r['max_drawdown']:.1%} exp {r['avg_exposure']:.3f} "
                  f"trades {int(r['n_trades'])} space {r['space_size']}", flush=True)
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    agg = d.groupby("variant").agg(
        seeds=("seed","count"), median_final=("final_equity","median"),
        min_final=("final_equity","min"), max_final=("final_equity","max"),
        median_cagr=("cagr","median"), median_sharpe=("sharpe","median"),
        median_sortino=("sortino","median"), median_dd=("max_drawdown","median"),
        median_exposure=("avg_exposure","median"), median_trades=("n_trades","median"),
        median_costs=("total_costs","median"), space=("space_size","max"),
    ).round(3)
    print("\n== batch 2, median of 3 seeds ==")
    print(agg.to_string())
    agg.to_csv("results/edge/variants2_summary.csv")
    print("\nEDGE2 DONE", flush=True)
