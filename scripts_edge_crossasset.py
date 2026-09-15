"""Freeze the accepted modification and test it on assets it never saw.

Same protocol as the previous cross-asset failure: the configuration the BTC
walk-forward chose most often is frozen and run unchanged, tiers are never
pooled, and each asset is compared against its own baselines. The previous run
found 0/4 unseen assets beating buy-and-hold; this re-runs that comparison with
the modification in place so the two are directly comparable.

Usage: EDGE_SPACE='{"signal_shape":[2.0]}' python scripts_edge_crossasset.py
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
from tradingagent.optimize import DEFAULT_SEARCH_SPACE, WalkForwardConfig, walk_forward

SPECS = {
    "BTC-USD": AssetSpec("BTC-USD", TRAINED, "coinbase", 365.0, "2015-01-01"),
    "ETH-USD": AssetSpec("ETH-USD", VALIDATION, "coinbase", 365.0, "2016-06-01"),
    "SPY": AssetSpec("SPY", UNSEEN, "nasdaq", 252.0, "2016-01-01"),
    "QQQ": AssetSpec("QQQ", UNSEEN, "nasdaq", 252.0, "2016-01-01"),
    "IWM": AssetSpec("IWM", UNSEEN, "nasdaq", 252.0, "2016-01-01"),
    "DIA": AssetSpec("DIA", UNSEEN, "nasdaq", 252.0, "2016-01-01"),
}
OVERRIDES = json.loads(os.environ.get("EDGE_SPACE", "{}"))
LABEL = os.environ.get("EDGE_LABEL", "modified")
SEED = int(os.environ.get("EDGE_SEED", "0"))

if __name__ == "__main__":
    ledger = ResearchLedger()
    space = {**DEFAULT_SEARCH_SPACE, **OVERRIDES}
    train = load_prices("BTC-USD", interval="1d", start="2015-01-01", source="coinbase")
    res = walk_forward(train, ExecutionConfig(initial_capital=100.0, record_trades=False),
                       WalkForwardConfig(n_candidates=64, seed=SEED, verbose=False), space,
                       ledger=ledger, label=f"edge_crossasset:{LABEL}")
    frozen = freeze_from_walk_forward(res, source=f"BTC daily walk-forward [{LABEL}]")
    print(f"[frozen:{LABEL}] {frozen.describe()}")
    print(f"[frozen:{LABEL}] overrides applied: {OVERRIDES}")

    datasets = {}
    for sym, spec in SPECS.items():
        try:
            datasets[sym] = load_prices(sym, interval="1d", start=spec.start, source=spec.source)
        except Exception as exc:
            print(f"[data] {sym} unavailable: {exc}")

    table = cross_asset_table(frozen, datasets, SPECS)
    table["label"] = LABEL
    pd.set_option("display.width", 250)
    cols = ["symbol","tier","final_equity","cagr","sharpe","sortino","max_drawdown","n_trades",
            "bl_buy_and_hold","beats_buy_hold","beats_sma_200"]
    print(f"\n== {LABEL}: frozen configuration across assets ==")
    print(table[[c for c in cols if c in table.columns]].round(3).to_string(index=False))
    print(f"\n== {LABEL}: verdict by tier ==")
    print(json.dumps(verdict(table), indent=1, default=str))
    out = f"results/edge/crossasset_{LABEL}.csv"
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")
