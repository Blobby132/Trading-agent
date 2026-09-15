"""Run the null suite against the daily BTC headline configuration.

Every replication re-runs the whole pipeline - search, fold loop, cost model -
so what comes back is the distribution of this machinery's output under each
null, which is the thing the real result must be compared against.
"""
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from tradingagent.data import load_prices
from tradingagent.engine import ExecutionConfig
from tradingagent.falsify import bootstrap_ohlcv, compare_to_null, flip_signs, shuffle_weights
from tradingagent.ledger import ResearchLedger
from tradingagent.optimize import WalkForwardConfig, walk_forward

SYMBOL = os.environ.get("NULL_SYMBOL", "BTC-USD")
REPS = int(os.environ.get("NULL_REPS", "20"))
N_CAND = int(os.environ.get("NULL_CANDIDATES", "64"))
BLOCK = int(os.environ.get("NULL_BLOCK", "21"))
OUT = f"results/sweeps/null_{SYMBOL}.jsonl"
NULLS = ("shuffled_signal", "sign_flipped", "bootstrapped_prices", "random_selection")


def _load():
    return load_prices(SYMBOL, interval="1d", start="2015-01-01", source="coinbase")


def _one(args):
    null, rep = args
    seed = 1000 + rep
    data = _load()
    base = ExecutionConfig(initial_capital=100.0, record_trades=False)
    wf = WalkForwardConfig(n_candidates=N_CAND, seed=0, verbose=False)
    kw = {}
    if null == "shuffled_signal":
        kw["weight_transform"] = shuffle_weights(BLOCK, seed)
    elif null == "sign_flipped":
        kw["weight_transform"] = flip_signs(BLOCK, seed)
    elif null == "bootstrapped_prices":
        data = bootstrap_ohlcv(data, block=BLOCK, rng=np.random.default_rng(seed))
    elif null == "random_selection":
        wf = WalkForwardConfig(n_candidates=N_CAND, seed=seed, verbose=False)
        kw["selection"] = "random"
    t0 = time.time()
    res = walk_forward(data, base, wf, None, **kw)
    s = res.stats()
    return {"null": null, "replication": rep, "seed": seed,
            "final_equity": s.get("final_equity"), "cagr": s.get("cagr"),
            "sharpe": s.get("sharpe"), "max_drawdown": s.get("max_drawdown"),
            "n_trades": s.get("n_trades"), "total_return": s.get("total_return"),
            "evaluations": res.n_evaluations, "seconds": time.time() - t0}


if __name__ == "__main__":
    # the observed result, run identically and recorded in the ledger
    ledger = ResearchLedger()
    data = _load()
    base = ExecutionConfig(initial_capital=100.0, record_trades=False)
    wf = WalkForwardConfig(n_candidates=N_CAND, seed=0, verbose=False)
    obs = walk_forward(data, base, wf, None, ledger=ledger,
                       label=f"observed:{SYMBOL}:1d", gross_twin=True)
    observed = obs.stats()
    print(f"[observed] ${observed['final_equity']:,.0f} Sharpe {observed['sharpe']:.3f} "
          f"CAGR {observed['cagr']:.1%} maxDD {observed['max_drawdown']:.1%} "
          f"({obs.n_evaluations} evaluations)", flush=True)

    jobs = [(n, r) for n in NULLS for r in range(REPS)]
    rows = []
    os.makedirs("results/sweeps", exist_ok=True)
    with open(OUT, "w") as fh, ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_one, j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            n, r = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                print(f"[null] FAILED {n} rep {r}: {type(exc).__name__}: {exc}", flush=True); continue
            rows.append(row); fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
            done += 1
            if done % 10 == 0:
                print(f"[null] {done}/{len(jobs)} done", flush=True)

    nulls = pd.DataFrame(rows)
    print("\n== null distributions ==", flush=True)
    print(nulls.groupby("null")[["final_equity", "sharpe", "max_drawdown", "n_trades"]]
          .agg(["median", "min", "max"]).round(3).to_string(), flush=True)
    for metric in ("sharpe", "final_equity"):
        print(f"\n== observed {metric} vs nulls ==", flush=True)
        print(compare_to_null(observed, nulls, metric=metric).round(4).to_string(index=False), flush=True)
    json.dump({"observed": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                            for k, v in observed.items()}},
              open("results/sweeps/null_observed.json", "w"), indent=1)
    print("\nNULL DONE", flush=True)
