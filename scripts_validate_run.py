"""Independent-market validation of BTC_FLOOR_050_CONTROL.

Validation only. One frozen parameter vector, applied unchanged everywhere. No
search, no per-asset tuning, no asset dropped for performing badly.
"""
import json, os
from dataclasses import replace

import numpy as np
import pandas as pd

from tradingagent.baselines import BASELINES, baseline_table, constant_exposure, run_baseline
from tradingagent.crossasset import FrozenConfig, apply_frozen
from tradingagent.data import load_prices
from tradingagent.engine import BacktestEngine, ExecutionConfig, buy_and_hold_equity
from tradingagent.execution import BASE_COST, HIGH_COST, LOW_COST, cost_scenario
from tradingagent.metrics import summarize
from tradingagent.optimize import sized_weight_for
from tradingagent.robustness import classify_regimes, regime_report

pd.set_option("display.width", 260)
FL = {"target_vol","atr_stop_mult","max_drawdown_stop","min_trade_frac","softmax_temp",
      "max_leverage","trend_floor","trend_tilt","accel_tilt","signal_shape","signal_deadband"}
raw = json.load(open("results/validation/frozen_params.json"))
PARAMS = {k: (v if isinstance(v, str) else (float(v) if k in FL else int(v))) for k, v in raw.items()}
FROZEN = FrozenConfig(params=PARAMS, source="BTC_FLOOR_050_CONTROL")

ASSETS = [
    ("BTC-USD", "coinbase", 365.0, "2015-01-01", "development"),
    ("ETH-USD", "coinbase", 365.0, "2016-06-01", "unseen-crypto"),
    ("SPY",     "nasdaq",   252.0, "2016-01-01", "unseen-equity"),
    ("QQQ",     "nasdaq",   252.0, "2016-01-01", "unseen-equity"),
    ("DIA",     "nasdaq",   252.0, "2016-01-01", "unseen-equity"),
    ("IWM",     "nasdaq",   252.0, "2016-01-01", "unseen-equity"),
]
BASELINE_NAMES = ("buy_and_hold", "forty_pct_invested", "half_invested",
                  "momentum_12m", "price_above_sma_200", "sma_50_200")


def exec_for(ppy, costs=None, cap=None):
    p = dict(PARAMS)
    cfg = ExecutionConfig(initial_capital=100.0, periods_per_year=ppy, record_trades=False)
    if costs is not None:
        cfg = replace(cfg, costs=costs)
    return cfg, p


def run_frozen(data, ppy, costs=None, cap=None, overrides=None):
    p = dict(PARAMS)
    if cap is not None:
        p["max_leverage"] = float(cap)
    if overrides:
        p.update(overrides)
    cfg = ExecutionConfig(initial_capital=100.0, periods_per_year=ppy, record_trades=False)
    if costs is not None:
        cfg = replace(cfg, costs=costs)
    return apply_frozen(FrozenConfig(params=p, source="frozen"), data, base_exec=cfg)


if __name__ == "__main__":
    data = {}
    for sym, src, ppy, start, tier in ASSETS:
        try:
            data[sym] = load_prices(sym, interval="1d", start=start, source=src)
            print(f"[data] {sym:<8} {len(data[sym]):>5} bars  "
                  f"{data[sym].index[0].date()} -> {data[sym].index[-1].date()}  [{tier}]")
        except Exception as e:
            print(f"[data] {sym} UNAVAILABLE: {e}")

    rows_main, rows_base, rows_regime, rows_cost, rows_perturb = [], [], [], [], []
    for sym, src, ppy, start, tier in ASSETS:
        if sym not in data: continue
        d = data[sym]
        # ---- Test A (exact frozen) and Test B (gross capped at 1.0) ----
        for label, cap in (("A_exact", None), ("B_cap1.0", 1.0)):
            s = run_frozen(d, ppy)if cap is None else run_frozen(d, ppy, cap=cap)
            rows_main.append({"symbol": sym, "tier": tier, "test": label,
                              "final_equity": s["final_equity"], "cagr": s["cagr"],
                              "sharpe": s["sharpe"], "sortino": s["sortino"],
                              "max_drawdown": s["max_drawdown"],
                              "avg_exposure": s["avg_gross_exposure"],
                              "turnover": s["turnover_per_year"], "n_trades": s["n_trades"],
                              "time_in_market": s["time_in_market"], "bars": s["bars"],
                              "total_costs": s.get("total_costs")})
        # ---- baselines on identical bars/costs/capital ----
        cfg = ExecutionConfig(initial_capital=100.0, periods_per_year=ppy, record_trades=False)
        bl = baseline_table(d, base_exec=cfg, names=BASELINE_NAMES, label=sym)
        bl["symbol"] = sym; bl["tier"] = tier
        rows_base.append(bl)
        # ---- exposure-matched constant, mechanically set to Test A's realised mean ----
        sA = run_frozen(d, ppy)
        mean_w = float(sA["avg_gross_exposure"])
        BASELINES["__matched"] = (lambda w: (lambda df: constant_exposure(df, w)))(mean_w)
        m = run_baseline("__matched", d, base_exec=cfg)
        rows_base.append(pd.DataFrame([{**{k: m.get(k) for k in
            ("final_equity","total_return","cagr","sharpe","sortino","max_drawdown",
             "calmar","n_trades","total_costs","time_in_market","bars")},
            "baseline": f"matched_{mean_w:.3f}", "label": sym, "symbol": sym, "tier": tier}]))
        # ---- regimes ----
        w, ec, rc = sized_weight_for(d, PARAMS, cfg)
        res = BacktestEngine(ec, rc).run(d, w)
        bench = buy_and_hold_equity(d, 100.0, costs=BASE_COST).reindex(res.equity.index).ffill()
        rr = regime_report(res.equity, classify_regimes(d["close"], periods_per_year=ppy),
                           periods_per_year=ppy, benchmark=bench)
        rr["symbol"] = sym; rr["tier"] = tier
        rows_regime.append(rr)
        # ---- costs ----
        for cname, cm in (("LOW", LOW_COST), ("BASE", BASE_COST), ("HIGH", HIGH_COST)):
            s = run_frozen(d, ppy, costs=cm)
            rows_cost.append({"symbol": sym, "tier": tier, "costs": cname,
                              "final_equity": s["final_equity"], "cagr": s["cagr"],
                              "sharpe": s["sharpe"], "max_drawdown": s["max_drawdown"]})
        # ---- parameter invariance, +/-20% on the two floor knobs only ----
        for pname, vals in (("trend_floor", (0.40, 0.50, 0.60)),
                            ("trend_floor_lookback", (202, 252, 302))):
            for v in vals:
                s = run_frozen(d, ppy, overrides={pname: v})
                rows_perturb.append({"symbol": sym, "tier": tier, "parameter": pname,
                                     "value": v, "final_equity": s["final_equity"],
                                     "sharpe": s["sharpe"], "cagr": s["cagr"]})
        print(f"[done] {sym}")

    main = pd.DataFrame(rows_main); main.to_csv("results/validation/main.csv", index=False)
    base_t = pd.concat(rows_base, ignore_index=True); base_t.to_csv("results/validation/baselines.csv", index=False)
    reg = pd.concat(rows_regime, ignore_index=True); reg.to_csv("results/validation/regimes.csv", index=False)
    cost = pd.DataFrame(rows_cost); cost.to_csv("results/validation/costs.csv", index=False)
    pert = pd.DataFrame(rows_perturb); pert.to_csv("results/validation/perturb.csv", index=False)

    print("\n== TEST A: exact frozen configuration ==")
    print(main[main.test=="A_exact"][["symbol","tier","final_equity","cagr","sharpe","sortino",
          "max_drawdown","avg_exposure","n_trades"]].round(3).to_string(index=False))
    print("\n== TEST B: same, gross capped at 1.0x (leverage-confound diagnostic) ==")
    print(main[main.test=="B_cap1.0"][["symbol","tier","final_equity","cagr","sharpe","sortino",
          "max_drawdown","avg_exposure"]].round(3).to_string(index=False))
    print("\nVALIDATION RUN DONE")
