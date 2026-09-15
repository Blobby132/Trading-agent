"""Diagnostic, not an optimisation.

HYPOTHESIS, stated before the run:

  The frozen BTC configuration carries target_vol=0.8 - an 80% annualised
  volatility target, which is reasonable for an asset that realises 60-80%. On
  an equity index realising ~18% it demands 4.4x exposure, capped by
  max_leverage=2.0, so the unseen assets were run permanently 2x levered.

  If the unseen-asset failure is mainly that risk-calibration artefact, then
  capping gross exposure at 1.0 - changing NOTHING about the signal - should
  move the unseen assets from losing money to roughly tracking buy-and-hold.

  If the signal itself does not generalise, removing leverage will cut the
  losses but the strategy will still underperform buy-and-hold materially.

One pre-specified change, one prediction, both outcomes named in advance. The
result is reported either way and is NOT accepted as an improvement - it is a
diagnostic that tells us which of two explanations to believe.
"""
import json
from dataclasses import replace

import pandas as pd

from tradingagent.crossasset import cross_asset_table, verdict, FrozenConfig
from tradingagent.data import load_prices
from scripts_crossasset import SPECS

if __name__ == "__main__":
    meta = json.load(open("results/sweeps/crossasset_frozen.json"))
    raw = meta["frozen_params"]
    def coerce(k, v):
        if k in ("weighting", "allocation"):
            return v
        try:
            return int(v) if float(v).is_integer() and k not in ("target_vol", "atr_stop_mult",
                                                                  "max_drawdown_stop", "min_trade_frac",
                                                                  "softmax_temp", "max_leverage") else float(v)
        except ValueError:
            return v
    params = {k: coerce(k, v) for k, v in raw.items()}

    datasets = {}
    for sym, spec in SPECS.items():
        try:
            datasets[sym] = load_prices(sym, interval="1d", start=spec.start, source=spec.source)
        except Exception as exc:
            print(f"[data] {sym} unavailable: {exc}")

    rows = []
    for cap in (2.0, 1.0):
        p = dict(params); p["max_leverage"] = cap
        frozen = FrozenConfig(params=p, source=f"BTC walk-forward, gross capped at {cap}x",
                              chosen_in_folds=meta["chosen_in_folds"], total_folds=meta["total_folds"])
        table = cross_asset_table(frozen, datasets, SPECS)
        table["max_leverage"] = cap
        rows.append(table)

    out = pd.concat(rows, ignore_index=True)
    pd.set_option("display.width", 250)
    cols = ["symbol", "tier", "max_leverage", "final_equity", "cagr", "sharpe",
            "max_drawdown", "avg_exposure" if "avg_exposure" in out.columns else "n_trades",
            "bl_buy_and_hold", "beats_buy_hold", "beats_sma_200"]
    print(out[[c for c in cols if c in out.columns]].round(3).to_string(index=False))
    out.to_csv("results/sweeps/crossasset_leverage_diagnostic.csv", index=False)
    print("\n== unseen tier, by leverage cap ==")
    for cap in (2.0, 1.0):
        sub = out[(out["tier"] == "unseen") & (out["max_leverage"] == cap)]
        print(f"  cap {cap}x: median final ${sub['final_equity'].median():,.0f}  "
              f"median Sharpe {sub['sharpe'].median():.3f}  "
              f"beats buy&hold {int(sub['beats_buy_hold'].sum())}/{len(sub)}  "
              f"worst DD {sub['max_drawdown'].min():.1%}")
    print("\nDIAG DONE")
