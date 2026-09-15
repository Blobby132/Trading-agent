"""Re-run the full robustness battery against an accepted modification.

Section 15 of the brief: a modification that improves in-sample performance but
fails robustness is not an improvement. This runs the same tests the baseline
faced - parameter perturbation, a cost ladder, regime decomposition and the
matched-exposure baselines - so before and after are directly comparable rather
than separately quoted.
"""
import json, os
from dataclasses import replace

import numpy as np
import pandas as pd

from tradingagent.baselines import BASELINES, constant_exposure, run_baseline
from tradingagent.crossasset import FrozenConfig, apply_frozen, freeze_from_walk_forward
from tradingagent.data import load_prices
from tradingagent.engine import ExecutionConfig
from tradingagent.execution import CostModel
from tradingagent.optimize import DEFAULT_SEARCH_SPACE, WalkForwardConfig, walk_forward
from tradingagent.robustness import classify_regimes, era_report, parameter_stability, regime_report

OVERRIDES = json.loads(os.environ.get("EDGE_SPACE", "{}"))
LABEL = os.environ.get("EDGE_LABEL", "modified")
SEED = int(os.environ.get("EDGE_SEED", "0"))
pd.set_option("display.width", 250)

if __name__ == "__main__":
    btc = load_prices("BTC-USD", interval="1d", start="2015-01-01", source="coinbase")
    base = ExecutionConfig(initial_capital=100.0, periods_per_year=365.0, record_trades=False)
    space = {**DEFAULT_SEARCH_SPACE, **OVERRIDES}
    res = walk_forward(btc, base, WalkForwardConfig(n_candidates=64, seed=SEED, verbose=False), space)
    frozen = freeze_from_walk_forward(res, source=f"BTC [{LABEL}]")
    traded = btc.loc[res.equity.index[0]:res.equity.index[-1]]
    print(f"[{LABEL}] frozen: {frozen.describe()}")

    # --- parameter perturbation -------------------------------------- #
    def evaluate(p):
        return apply_frozen(FrozenConfig(params=p, source="perturb"), traded, base_exec=base)
    stab = parameter_stability(evaluate, frozen.params, metric="sharpe",
                               relative=0.20, n_points=5, verbose=False)
    spikes = stab[stab["verdict"].astype(str).str.startswith("SPIKE")]
    print(f"\n[{LABEL}] parameter stability: median {stab['stability'].median():.3f}, "
          f"{len(spikes)} spike(s) {list(spikes['parameter'])}, "
          f"{int((stab['verdict']=='plateau').sum())}/{len(stab)} plateau")
    stab.to_csv(f"results/edge/stability_{LABEL}.csv", index=False)

    # --- cost ladder --------------------------------------------------- #
    rows = []
    for bps in (0, 4, 15, 25, 35, 50, 75, 100, 150):
        cm = CostModel(fee_bps=bps*0.667, half_spread_bps=bps*0.1665,
                       slippage_bps=bps*0.1665, impact_bps_at_full=0.0, name=f"{bps}")
        s = apply_frozen(frozen, traded, base_exec=replace(base, costs=cm))
        rows.append({"bps_per_side": bps, "final_equity": s["final_equity"],
                     "sharpe": s["sharpe"], "max_drawdown": s["max_drawdown"]})
    ladder = pd.DataFrame(rows)
    ladder.to_csv(f"results/edge/cost_ladder_{LABEL}.csv", index=False)
    print(f"\n[{LABEL}] cost ladder:")
    print(ladder.round(3).to_string(index=False))

    # --- matched-exposure baselines ------------------------------------ #
    s = res.stats()
    mean_w = float(s["avg_gross_exposure"])
    BASELINES["matched"] = lambda df: constant_exposure(df, mean_w)
    matched = run_baseline("matched", traded, base_exec=base)
    c40 = run_baseline("forty_pct_invested", traded, base_exec=base)
    mom = run_baseline("momentum_12m", traded, base_exec=base)
    bh = run_baseline("buy_and_hold", traded, base_exec=base)
    print(f"\n[{LABEL}] walk-forward OOS: ${s['final_equity']:,.0f} CAGR {s['cagr']:.1%} "
          f"Sharpe {s['sharpe']:.3f} Sortino {s['sortino']:.3f} maxDD {s['max_drawdown']:.1%} "
          f"exposure {mean_w:.3f} trades {int(s['n_trades'])}")
    for name, b in (("matched constant", matched), ("constant 40%", c40),
                    ("momentum_12m", mom), ("buy_and_hold", bh)):
        print(f"  vs {name:<18} ${b['final_equity']:>9,.0f}  Sharpe {b['sharpe']:.3f}  "
              f"Sortino {b['sortino']:.3f}  ->  equity {s['final_equity']/b['final_equity']-1:+7.1%}  "
              f"Sharpe {s['sharpe']-b['sharpe']:+.3f}  Sortino {s['sortino']-b['sortino']:+.3f}")

    # --- regimes ------------------------------------------------------- #
    from tradingagent.engine import buy_and_hold_equity
    from tradingagent.execution import BASE_COST
    bench = buy_and_hold_equity(traded, 100.0, costs=BASE_COST).reindex(res.equity.index).ffill()
    reg = regime_report(res.equity, classify_regimes(traded["close"], periods_per_year=365.0),
                        periods_per_year=365.0, benchmark=bench)
    trend = reg[reg["dimension"] == "trend"]
    print(f"\n[{LABEL}] regimes with positive excess: "
          f"{int((trend['excess']>0).sum())}/{len(trend)} {list(trend[trend['excess']>0]['regime'])}")
    print(trend[["regime","bars","total_return","benchmark_return","excess","sharpe"]].round(3).to_string(index=False))
    reg.to_csv(f"results/edge/regime_{LABEL}.csv", index=False)
    json.dump({"label": LABEL, "overrides": OVERRIDES, "stats": {k: float(v) for k, v in s.items()
               if isinstance(v, (int, float, np.floating))},
               "matched_exposure": mean_w,
               "stability_median": float(stab["stability"].median()),
               "n_spikes": int(len(spikes))},
              open(f"results/edge/robustness_{LABEL}.json", "w"), indent=1)
    print(f"\nROBUSTNESS DONE {LABEL}")
