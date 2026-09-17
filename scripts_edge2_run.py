"""Edge phase 2 experiments, against BTC_FLOOR_050_CONTROL.

Strict budget. Two hypotheses survived a causal screen of eight candidate
conditioning variables, and each is tested alone before any combination.
Every variant fixes its knobs; the underlying 2,304-configuration search is
unchanged, so the optimiser cannot hunt for the tilt that flatters it.
"""
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from tradingagent.control import CONTROL_OVERRIDES
from tradingagent.data import load_prices
from tradingagent.engine import ExecutionConfig
from tradingagent.optimize import DEFAULT_SEARCH_SPACE, WalkForwardConfig, space_size, walk_forward

# every variant carries the protected floor
VARIANTS = json.loads(os.environ.get("EDGE2_VARIANTS", json.dumps({
    "CONTROL":            {},
    "E1_trend0.25":       {"trend_tilt": [0.25]},
    "E2_trend0.50":       {"trend_tilt": [0.50]},
    "E3_accel0.25":       {"accel_tilt": [0.25]},
    "E4_accel0.50":       {"accel_tilt": [0.50]},
})))
SEEDS = [int(s) for s in os.environ.get("EDGE2_SEEDS", "0,1,2,3").split(",")]
OUT = os.environ.get("EDGE2_OUT", "results/edge2/variants.jsonl")


def _one(args):
    name, seed = args
    btc = load_prices("BTC-USD", interval="1d", start="2015-01-01", source="coinbase")
    space = {**DEFAULT_SEARCH_SPACE, **CONTROL_OVERRIDES, **VARIANTS[name]}
    base = ExecutionConfig(initial_capital=100.0, periods_per_year=365.0, record_trades=False)
    t0 = time.time()
    res = walk_forward(btc, base, WalkForwardConfig(n_candidates=64, seed=seed, verbose=False), space)
    s = res.stats()
    return {"variant": name, "seed": seed, "final_equity": s["final_equity"],
            "cagr": s["cagr"], "sharpe": s["sharpe"], "sortino": s["sortino"],
            "max_drawdown": s["max_drawdown"], "n_trades": s["n_trades"],
            "turnover": s["turnover_per_year"], "avg_exposure": s["avg_gross_exposure"],
            "total_costs": s.get("total_costs"), "bars": s["bars"],
            "evaluations": res.n_evaluations, "space": space_size(space),
            "seconds": time.time() - t0}


if __name__ == "__main__":
    jobs = [(n, s) for n in VARIANTS for s in SEEDS]
    os.makedirs("results/edge2", exist_ok=True)
    rows = []
    with open(OUT, "w") as fh, ProcessPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_one, j): j for j in jobs}
        done = 0
        for f in as_completed(futs):
            n, s = futs[f]
            try:
                r = f.result()
            except Exception as e:
                print(f"[e2] FAILED {n}/{s}: {type(e).__name__}: {e}", flush=True); continue
            rows.append(r); fh.write(json.dumps(r, default=str) + "\n"); fh.flush()
            done += 1
            print(f"[e2 {done}/{len(jobs)}] {n} seed {s}: ${r['final_equity']:,.0f} "
                  f"CAGR {r['cagr']:.1%} Sh {r['sharpe']:.3f} So {r['sortino']:.3f} "
                  f"DD {r['max_drawdown']:.1%} exp {r['avg_exposure']:.3f} "
                  f"tr {int(r['n_trades'])}", flush=True)

    d = pd.DataFrame(rows); pd.set_option("display.width", 250)
    print("\n== medians ==")
    print(d.groupby("variant").agg(
        seeds=("seed","count"), median_final=("final_equity","median"),
        min_final=("final_equity","min"), max_final=("final_equity","max"),
        median_cagr=("cagr","median"), median_sharpe=("sharpe","median"),
        median_sortino=("sortino","median"), median_dd=("max_drawdown","median"),
        median_exposure=("avg_exposure","median"), median_trades=("n_trades","median"),
    ).round(3).to_string())

    if "CONTROL" in set(d["variant"]):
        eq = d.pivot(index="seed", columns="variant", values="final_equity")
        sh = d.pivot(index="seed", columns="variant", values="sharpe")
        ex = d.pivot(index="seed", columns="variant", values="avg_exposure")
        print("\n== paired vs CONTROL (same seed, same market path) ==")
        for v in eq.columns:
            if v == "CONTROL": continue
            de = (eq[v]/eq["CONTROL"] - 1).dropna()
            ds = (sh[v] - sh["CONTROL"]).dropna()
            dx = (ex[v]/ex["CONTROL"] - 1).dropna()
            t = de.mean()/(de.std(ddof=1)/np.sqrt(len(de))) if len(de) > 1 else np.nan
            print(f"  {v:<16} equity {de.mean():+7.2%} (wins {int((de>0).sum())}/{len(de)}, "
                  f"t={t:+.2f})  Sharpe {ds.mean():+.3f} (wins {int((ds>0).sum())}/{len(ds)})  "
                  f"exposure {dx.mean():+.1%}")
    print("\nEDGE2 DONE", flush=True)
