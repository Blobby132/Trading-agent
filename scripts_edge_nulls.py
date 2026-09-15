"""Null retest for the modified strategy (brief section 15).

Two nulls, the two that carried the evidence last time: block-shuffled timing,
and the drift-controlled bootstrap - the latter differencing the strategy
against buy-and-hold on the SAME synthetic path, because a circular block
bootstrap makes every path easier and absolute Sharpes are not comparable
across that gap.
"""
import json, os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from tradingagent.baselines import run_baseline
from tradingagent.data import load_prices
from tradingagent.engine import ExecutionConfig
from tradingagent.falsify import bootstrap_ohlcv, compare_to_null, drift_controlled_comparison, shuffle_weights
from tradingagent.optimize import DEFAULT_SEARCH_SPACE, WalkForwardConfig, walk_forward

OVERRIDES = json.loads(os.environ.get("EDGE_SPACE", '{"trend_floor":[0.50]}'))
LABEL = os.environ.get("EDGE_LABEL", "floor050")
REPS = int(os.environ.get("EDGE_REPS", "20"))
BLOCK = 21


def _one(args):
    null, rep = args
    seed = 2000 + rep
    data = load_prices("BTC-USD", interval="1d", start="2015-01-01", source="coinbase")
    base = ExecutionConfig(initial_capital=100.0, record_trades=False)
    space = {**DEFAULT_SEARCH_SPACE, **OVERRIDES}
    wf = WalkForwardConfig(n_candidates=64, seed=0, verbose=False)
    kw = {}
    bh_sharpe = np.nan
    if null == "shuffled_signal":
        kw["weight_transform"] = shuffle_weights(BLOCK, seed)
    else:
        data = bootstrap_ohlcv(data, block=BLOCK, rng=np.random.default_rng(seed))
        bh = run_baseline("buy_and_hold", data.iloc[740:],
                          base_exec=ExecutionConfig(initial_capital=100.0, periods_per_year=365.0,
                                                    record_trades=False))
        bh_sharpe = bh["sharpe"]
    res = walk_forward(data, base, wf, space, **kw)
    s = res.stats()
    return {"null": null, "rep": rep, "seed": seed, "sharpe": s["sharpe"],
            "final_equity": s["final_equity"], "buyhold_sharpe": bh_sharpe}


if __name__ == "__main__":
    jobs = [(n, r) for n in ("shuffled_signal", "bootstrapped_prices") for r in range(REPS)]
    rows = []
    out = f"results/edge/nulls_{LABEL}.jsonl"
    with open(out, "w") as fh, ProcessPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_one, j): j for j in jobs}
        done = 0
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:
                print(f"[null] FAILED {futs[f]}: {e}", flush=True); continue
            rows.append(r); fh.write(json.dumps(r, default=str) + "\n"); fh.flush()
            done += 1
            if done % 10 == 0:
                print(f"[null] {done}/{len(jobs)}", flush=True)
    d = pd.DataFrame(rows)
    obs = json.load(open(f"results/edge/robustness_{LABEL}.json"))["stats"]
    print(f"\nobserved: Sharpe {obs['sharpe']:.3f}  final ${obs['final_equity']:,.0f}")
    print("\n== raw null comparison ==")
    print(compare_to_null(obs, d, metric="sharpe").round(4).to_string(index=False))

    boot = d[d["null"] == "bootstrapped_prices"].dropna(subset=["buyhold_sharpe"])
    real_bh = run_baseline("buy_and_hold",
                           load_prices("BTC-USD", interval="1d", start="2015-01-01",
                                       source="coinbase").loc["2017-07-29":"2026-09-12"],
                           base_exec=ExecutionConfig(initial_capital=100.0, periods_per_year=365.0,
                                                     record_trades=False))["sharpe"]
    edge = obs["sharpe"] - real_bh
    dc = drift_controlled_comparison(boot["sharpe"], boot["buyhold_sharpe"], edge)
    print(f"\n== DRIFT-CONTROLLED (strategy minus buy-and-hold on the same path) ==")
    print(f"  real edge vs buy-and-hold: {edge:+.4f}  (strategy {obs['sharpe']:.3f} - bh {real_bh:.3f})")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in dc.items()}, indent=1))
    json.dump({"raw": compare_to_null(obs, d, metric="sharpe").to_dict("records"),
               "drift_controlled": dc, "real_edge": edge},
              open(f"results/edge/null_summary_{LABEL}.json", "w"), indent=1, default=str)
    print("\nEDGE NULLS DONE", flush=True)
