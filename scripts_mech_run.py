"""Pre-registered mechanism experiments. Walk-forward, BTC only at this stage.

H1 - the cross-asset mechanism is volatility targeting, not the trend floor.
     Ablation: removing the floor costs BTC -57.8% and ETH +1.2%. If the floor
     were the travelling mechanism, ETH would depend on it. It does not.

H2 - the drawdown kill switch destroys value on crypto.
     Ablation: removing it gains +27.4% on BTC and +26.2% on ETH while costing
     -4.6% on SPY. A component that consistently costs money on the assets where
     the strategy works is a simplification candidate, not a risk control.

ETH is NOT used to select anything here. Whatever survives on BTC is frozen and
then run on ETH once.
"""
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from tradingagent.control import CONTROL_OVERRIDES
from tradingagent.data import load_prices
from tradingagent.engine import ExecutionConfig
from tradingagent.optimize import DEFAULT_SEARCH_SPACE, WalkForwardConfig, walk_forward

VARIANTS = json.loads(os.environ.get("MECH_VARIANTS", json.dumps({
    "M0_control":            {},
    "M1_no_killswitch":      {"max_drawdown_stop": [0.0]},
    "M2_no_floor":           {"trend_floor": [0.0]},
    "M3_no_floor_no_kill":   {"trend_floor": [0.0], "max_drawdown_stop": [0.0]},
})))
SEEDS = [int(s) for s in os.environ.get("MECH_SEEDS", "0,1,2,3").split(",")]
SYMBOL = os.environ.get("MECH_SYMBOL", "BTC-USD")
SOURCE = os.environ.get("MECH_SOURCE", "coinbase")
PPY = float(os.environ.get("MECH_PPY", "365"))
START = os.environ.get("MECH_START", "2015-01-01")
OUT = os.environ.get("MECH_OUT", "results/mechanism/walkforward.jsonl")


def _one(args):
    name, seed = args
    d = load_prices(SYMBOL, interval="1d", start=START, source=SOURCE)
    space = {**DEFAULT_SEARCH_SPACE, **CONTROL_OVERRIDES, **VARIANTS[name]}
    base = ExecutionConfig(initial_capital=100.0, periods_per_year=PPY, record_trades=False)
    t0 = time.time()
    res = walk_forward(d, base, WalkForwardConfig(n_candidates=64, seed=seed, verbose=False), space)
    s = res.stats()
    return {"variant": name, "symbol": SYMBOL, "seed": seed,
            "final_equity": s["final_equity"], "cagr": s["cagr"], "sharpe": s["sharpe"],
            "sortino": s["sortino"], "max_drawdown": s["max_drawdown"],
            "n_trades": s["n_trades"], "turnover": s["turnover_per_year"],
            "avg_exposure": s["avg_gross_exposure"], "total_costs": s.get("total_costs"),
            "bars": s["bars"], "evaluations": res.n_evaluations, "seconds": time.time()-t0}


if __name__ == "__main__":
    jobs = [(n, s) for n in VARIANTS for s in SEEDS]
    os.makedirs("results/mechanism", exist_ok=True)
    rows = []
    with open(OUT, "w") as fh, ProcessPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_one, j): j for j in jobs}
        done = 0
        for f in as_completed(futs):
            n, s = futs[f]
            try:
                r = f.result()
            except Exception as e:
                print(f"[mech] FAILED {n}/{s}: {type(e).__name__}: {e}", flush=True); continue
            rows.append(r); fh.write(json.dumps(r, default=str)+"\n"); fh.flush()
            done += 1
            print(f"[mech {done}/{len(jobs)}] {SYMBOL} {n} seed {s}: ${r['final_equity']:,.0f} "
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
    if "M0_control" in set(d["variant"]):
        eq = d.pivot(index="seed", columns="variant", values="final_equity")
        sh = d.pivot(index="seed", columns="variant", values="sharpe")
        so = d.pivot(index="seed", columns="variant", values="sortino")
        dd = d.pivot(index="seed", columns="variant", values="max_drawdown")
        print("\n== paired vs M0_control (same seed, same path) ==")
        for v in eq.columns:
            if v == "M0_control": continue
            de = (eq[v]/eq["M0_control"]-1).dropna()
            t = de.mean()/(de.std(ddof=1)/np.sqrt(len(de))) if len(de)>1 else np.nan
            print(f"  {v:<22} equity {de.mean():+7.2%} (wins {int((de>0).sum())}/{len(de)}, t={t:+.2f})  "
                  f"Sharpe {(sh[v]-sh['M0_control']).mean():+.3f}  "
                  f"Sortino {(so[v]-so['M0_control']).mean():+.3f}  "
                  f"maxDD {(dd[v]-dd['M0_control']).mean():+.3f}")
    print("\nMECH DONE", flush=True)
