"""Step 1: freeze the exact BTC parameter vector and fingerprint the run.

The vector is taken from the control's own walk-forward - the configuration it
selected most often across folds - and is then never re-derived. Every asset in
this phase sees this exact vector.
"""
import hashlib, json, os, subprocess, sys

import pandas as pd

from tradingagent.control import CONTROL_METRICS, CONTROL_NAME, CONTROL_OVERRIDES, CONTROL_SPEC
from tradingagent.crossasset import freeze_from_walk_forward
from tradingagent.data import load_prices
from tradingagent.engine import ExecutionConfig
from tradingagent.optimize import DEFAULT_SEARCH_SPACE, WalkForwardConfig, walk_forward

SEED = 0

def sha_of(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(open(p, "rb").read())
    return h.hexdigest()[:16]

if __name__ == "__main__":
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())
    btc = load_prices("BTC-USD", interval="1d", start="2015-01-01", source="coinbase")
    space = {**DEFAULT_SEARCH_SPACE, **CONTROL_OVERRIDES}
    base = ExecutionConfig(initial_capital=100.0, periods_per_year=365.0, record_trades=False)
    res = walk_forward(btc, base, WalkForwardConfig(n_candidates=64, seed=SEED, verbose=False), space)
    frozen = freeze_from_walk_forward(res, source=f"{CONTROL_NAME} walk-forward, seed {SEED}")
    s = res.stats()

    code = ["tradingagent/agent.py", "tradingagent/risk.py", "tradingagent/engine.py",
            "tradingagent/execution.py", "tradingagent/optimize.py", "tradingagent/control.py"]
    fp = {
        "experiment_id": "INDEPENDENT_EDGE_VALIDATION",
        "commit_sha": commit,
        "working_tree_dirty": dirty,
        "code_fingerprint_sha256_16": sha_of(code),
        "code_files_hashed": code,
        "control_name": CONTROL_NAME,
        "control_spec": CONTROL_SPEC,
        "control_metrics_as_recorded": CONTROL_METRICS,
        "frozen_params": {k: str(v) for k, v in frozen.params.items()},
        "frozen_source": frozen.source,
        "frozen_chosen_in_folds": f"{frozen.chosen_in_folds}/{frozen.total_folds}",
        "seed": SEED,
        "btc_regenerated_now": {
            "final_equity": float(s["final_equity"]), "cagr": float(s["cagr"]),
            "sharpe": float(s["sharpe"]), "sortino": float(s["sortino"]),
            "max_drawdown": float(s["max_drawdown"]), "avg_exposure": float(s["avg_gross_exposure"]),
            "n_trades": int(s["n_trades"]), "bars": int(s["bars"]),
            "traded_start": str(res.equity.index[0].date()),
            "traded_end": str(res.equity.index[-1].date()),
        },
        "execution_timing": "decision at close of bar t, fill at open of bar t+1",
        "costs_base_bps_per_side": 15.0,
        "pandas": pd.__version__, "python": sys.version.split()[0],
    }
    os.makedirs("results/validation", exist_ok=True)
    json.dump(fp, open("results/validation/fingerprint.json", "w"), indent=1)
    json.dump({k: frozen.params[k] for k in sorted(frozen.params)},
              open("results/validation/frozen_params.json", "w"), indent=1)
    print(json.dumps({k: fp[k] for k in ("commit_sha","working_tree_dirty",
          "code_fingerprint_sha256_16","frozen_chosen_in_folds","btc_regenerated_now")},
          indent=1))
    print("\nfrozen parameter vector:")
    for k in sorted(frozen.params):
        print(f"  {k:<22} {frozen.params[k]}")
