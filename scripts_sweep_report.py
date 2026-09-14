"""Turn the sweep result files into the tables in docs/INTERVAL_FREQUENCY.md.

Generated rather than hand-copied, so the report cannot drift from the runs.
"""
import json, sys, glob
import numpy as np
import pandas as pd

from tradingagent.metrics import deflated_sharpe
from tradingagent.timescale import CRYPTO_SCALES


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return pd.DataFrame(rows)


def money(x):
    return f"${x:,.0f}" if np.isfinite(x) else "n/a"


def crypto_table(df):
    df = df.copy()
    order = {"1d": 0, "6h": 1, "1h": 2, "15m": 3}
    df = df.sort_values(["interval", "every"], key=lambda c: c.map(order) if c.name == "interval" else c)
    total_trials = int(df["evaluations"].sum())
    df["dsr_cell"] = [
        deflated_sharpe(r.sharpe, int(r.evaluations), int(r.bars),
                        CRYPTO_SCALES[r.interval].bars_per_year)
        for r in df.itertuples()
    ]
    df["dsr_sweep"] = [
        deflated_sharpe(r.sharpe, total_trials, int(r.bars),
                        CRYPTO_SCALES[r.interval].bars_per_year)
        for r in df.itertuples()
    ]
    lines = [
        "| interval | rebalance | trades/day (intended → actual) | net final | gross final | "
        "cost drag | cost as % of gross profit | CAGR | Sharpe | maxDD | DSR (cell) | DSR (sweep) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        cg = r.cost_drag_pct_of_gross_profit
        lines.append(
            f"| {r.interval} | every {r.every} | {r.intended_trades_day:.2g} → {r.actual_trades_day:.2f} | "
            f"**{money(r.final_equity)}** | {money(r.gross_final_equity)} | "
            f"{r.cost_drag_pct_capital:.0%} of stake | "
            f"{(f'{cg:.0%}' if np.isfinite(cg) else 'n/a')} | "
            f"{r.cagr:.1%} | {r.sharpe:.2f} | {r.max_drawdown:.0%} | "
            f"{r.dsr_cell:.2f} | {r.dsr_sweep:.2f} |"
        )
    return "\n".join(lines), total_trials, df


def stock_table(df):
    df = df.sort_values("rebalance_every")
    lines = [
        "| rebalance | fills/day | net final | gross final | cost drag | CAGR | Sharpe | maxDD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        lines.append(
            f"| every {r.rebalance_every} bar{'s' if r.rebalance_every > 1 else ''} | "
            f"{r.trades_per_day:.1f} | **{money(r.final_equity)}** | {money(r.gross_final_equity)} | "
            f"{r.cost_drag_pct_capital:.0%} of stake | {r.cagr:.1%} | {r.sharpe:.2f} | "
            f"{r.max_drawdown:.0%} |"
        )
    return "\n".join(lines), df


if __name__ == "__main__":
    for path in sorted(glob.glob("results/sweeps/*.jsonl")):
        df = load(path)
        if df.empty:
            print(f"\n## {path}\n(empty)\n"); continue
        print(f"\n## {path}  ({len(df)} cells)\n")
        if "interval" in df.columns:
            table, trials, full = crypto_table(df)
            print(table)
            print(f"\ntotal configurations evaluated across the sweep: {trials:,}\n")
            full.to_csv(path.replace(".jsonl", "_table.csv"), index=False)
        else:
            table, full = stock_table(df)
            print(table)
            full.to_csv(path.replace(".jsonl", "_table.csv"), index=False)
