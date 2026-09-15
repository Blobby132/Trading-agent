"""Freeze the configuration chosen on BTC, then run it unchanged elsewhere.

Tiers are kept apart deliberately. The trained asset's number is in-sample with
respect to parameter choice and is reported for reference only; the unseen
equity ETFs are the tier that carries the evidence.
"""
import json, os

import pandas as pd

from tradingagent.crossasset import (
    TRAINED, UNSEEN, VALIDATION, AssetSpec, cross_asset_table,
    freeze_from_walk_forward, verdict,
)
from tradingagent.data import load_prices
from tradingagent.engine import ExecutionConfig
from tradingagent.ledger import ResearchLedger
from tradingagent.optimize import WalkForwardConfig, walk_forward

N_CAND = int(os.environ.get("XA_CANDIDATES", "64"))
SEED = int(os.environ.get("XA_SEED", "0"))

SPECS = {
    "BTC-USD": AssetSpec("BTC-USD", TRAINED, "coinbase", 365.0, "2015-01-01",
                         "parameters were selected on this series"),
    "ETH-USD": AssetSpec("ETH-USD", VALIDATION, "coinbase", 365.0, "2016-06-01",
                         "same asset class, highly correlated with BTC"),
    "SPY": AssetSpec("SPY", UNSEEN, "nasdaq", 252.0, "2016-01-01", "US large cap index"),
    "QQQ": AssetSpec("QQQ", UNSEEN, "nasdaq", 252.0, "2016-01-01", "US tech index"),
    "IWM": AssetSpec("IWM", UNSEEN, "nasdaq", 252.0, "2016-01-01", "US small cap index"),
    "DIA": AssetSpec("DIA", UNSEEN, "nasdaq", 252.0, "2016-01-01", "US industrials index"),
}

if __name__ == "__main__":
    ledger = ResearchLedger()
    train = load_prices("BTC-USD", interval="1d", start="2015-01-01", source="coinbase")
    wf = WalkForwardConfig(n_candidates=N_CAND, seed=SEED, verbose=False)
    res = walk_forward(train, ExecutionConfig(initial_capital=100.0, record_trades=False),
                       wf, None, ledger=ledger, label="crossasset:train:BTC-USD")
    frozen = freeze_from_walk_forward(res, source="BTC-USD daily 2015-2026 walk-forward")
    print(f"[frozen] {frozen.describe()}", flush=True)
    print(f"[frozen] params: {json.dumps(frozen.params, default=str, sort_keys=True)}", flush=True)

    datasets = {}
    for sym, spec in SPECS.items():
        try:
            datasets[sym] = load_prices(sym, interval="1d", start=spec.start, source=spec.source)
            print(f"[data] {sym}: {len(datasets[sym])} bars "
                  f"{datasets[sym].index[0].date()} -> {datasets[sym].index[-1].date()}", flush=True)
        except Exception as exc:
            print(f"[data] {sym} UNAVAILABLE: {type(exc).__name__}: {exc}", flush=True)

    table = cross_asset_table(frozen, datasets, SPECS)
    pd.set_option("display.width", 250)
    cols = ["symbol", "tier", "bars", "final_equity", "cagr", "sharpe", "max_drawdown",
            "n_trades", "bl_buy_and_hold", "bl_price_above_sma_200", "bl_momentum_12m",
            "beats_buy_hold", "beats_sma_200"]
    print("\n== frozen configuration across assets ==")
    print(table[[c for c in cols if c in table.columns]].round(3).to_string(index=False))
    print("\n== verdict by tier ==")
    print(json.dumps(verdict(table), indent=1, default=str))
    table.to_csv("results/sweeps/crossasset.csv", index=False)
    json.dump({"frozen_params": {k: str(v) for k, v in frozen.params.items()},
               "source": frozen.source, "chosen_in_folds": frozen.chosen_in_folds,
               "total_folds": frozen.total_folds},
              open("results/sweeps/crossasset_frozen.json", "w"), indent=1)
    print("\nCROSSASSET DONE", flush=True)
