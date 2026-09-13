"""Paper trading: run the strategy forward and find out whether it was real.

A backtest is a claim about the future. The only way to test it is to run the
thing forward without money and compare what actually happened against what the
backtest said would happen. That is what this module is for, and the comparison
matters more than the P&L: a paper account that makes money while behaving
nothing like its backtest has told you the backtest is wrong, not that the
strategy works.

State lives in one JSON file, so the account survives restarts and the whole
history is inspectable in a text editor. The workflow is three commands:

    python -m tradingagent.paper init    --capital 100
    python -m tradingagent.paper rebalance          # once a month
    python -m tradingagent.paper report             # any time

Nothing here places an order. ``rebalance`` prints the orders and records them
as filled at the next open, which is the same convention the backtest uses, so
the two remain comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .cross_section import PortfolioRules, SingleFeatureRanker, scores_to_weights
from .features import feature_panel
from .universe import UNIVERSES, Panel, load_panel

DEFAULT_STATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "paper_account.json"
)


@dataclass
class Position:
    symbol: str
    shares: float
    avg_price: float

    def value(self, price: float) -> float:
        return self.shares * price


@dataclass
class PaperAccount:
    """A paper portfolio, its history, and the backtest it is being judged against."""

    capital: float = 100.0
    cash: float = 100.0
    positions: Dict[str, Position] = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)      # one row per mark
    orders: List[dict] = field(default_factory=list)       # every fill
    universe: str = "us_large_cap"
    source: str = "yahoo"
    strategy: str = "mom_12_1"
    top_frac: float = 0.10
    max_weight: float = 0.15
    started: str = ""
    expected_annual_return: float = 0.0    # what the backtest said, for comparison
    expected_annual_vol: float = 0.0

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str = DEFAULT_STATE) -> "PaperAccount":
        with open(path) as fh:
            raw = json.load(fh)
        raw["positions"] = {
            k: Position(**v) for k, v in raw.get("positions", {}).items()
        }
        return cls(**raw)

    def save(self, path: str = DEFAULT_STATE) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = asdict(self)
        payload["positions"] = {k: asdict(v) for k, v in self.positions.items()}
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=1)

    # ------------------------------------------------------------------ #
    def equity(self, prices: pd.Series) -> float:
        held = sum(
            pos.value(float(prices.get(sym, pos.avg_price))) for sym, pos in self.positions.items()
        )
        return float(self.cash + held)

    def target_basket(self, panel: Panel, *, periods_per_year: float = 252.0) -> pd.Series:
        """Today's target weights from the fixed ranker."""
        feats = feature_panel(panel, periods_per_year=periods_per_year)
        rules = PortfolioRules(
            long_only=True, top_frac=self.top_frac, gross=1.0,
            max_weight=self.max_weight, rebalance_every=1,
        )
        weights = scores_to_weights(SingleFeatureRanker(self.strategy).score(feats), rules)
        return weights.iloc[-1]

    def plan_orders(
        self, panel: Panel, *, min_trade_frac: float = 0.10
    ) -> pd.DataFrame:
        """What to buy and sell to reach the target basket.

        Sized against the *last close*, executed at the next open - the same
        timing the backtest uses, so the two stay comparable. A full exit is
        never suppressed as dust.
        """
        prices = panel.close.iloc[-1]
        equity = self.equity(prices)
        target = self.target_basket(panel)

        rows = []
        universe = set(target.index) | set(self.positions)
        for sym in sorted(universe):
            price = float(prices.get(sym, np.nan))
            if not np.isfinite(price) or price <= 0:
                continue
            want_weight = float(target.get(sym, 0.0))
            want_shares = want_weight * equity / price
            have_shares = self.positions[sym].shares if sym in self.positions else 0.0
            delta = want_shares - have_shares
            if abs(delta) < 1e-12:
                continue
            position_ref = max(abs(want_shares), abs(have_shares)) * price
            full_exit = want_shares == 0.0 and have_shares != 0.0
            if not full_exit and abs(delta) * price < min_trade_frac * position_ref:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "side": "BUY" if delta > 0 else "SELL",
                    "shares": abs(delta),
                    "price": price,
                    "notional": abs(delta) * price,
                    "target_weight": want_weight,
                    "current_weight": have_shares * price / equity if equity else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def apply_orders(
        self, orders: pd.DataFrame, prices: pd.Series, as_of: pd.Timestamp, *, cost_bps: float = 8.0
    ) -> None:
        """Record fills. Costs are charged the same way the backtest charges them."""
        for _, row in orders.iterrows():
            sym = row["symbol"]
            price = float(prices.get(sym, row["price"]))
            shares = float(row["shares"]) * (1 if row["side"] == "BUY" else -1)
            cost = abs(shares) * price * cost_bps / 1e4
            self.cash -= shares * price + cost

            existing = self.positions.get(sym)
            new_shares = (existing.shares if existing else 0.0) + shares
            if abs(new_shares) < 1e-9:
                self.positions.pop(sym, None)
            elif existing and np.sign(new_shares) == np.sign(existing.shares) and shares > 0:
                total_cost = existing.avg_price * existing.shares + price * shares
                self.positions[sym] = Position(sym, new_shares, total_cost / new_shares)
            elif existing:
                self.positions[sym] = Position(sym, new_shares, existing.avg_price)
            else:
                self.positions[sym] = Position(sym, new_shares, price)

            self.orders.append(
                {
                    "at": as_of.isoformat(),
                    "symbol": sym,
                    "side": row["side"],
                    "shares": float(row["shares"]),
                    "price": price,
                    "cost": cost,
                }
            )

    def mark(self, panel: Panel) -> dict:
        """Record today's equity. Idempotent per date."""
        as_of = panel.index[-1]
        prices = panel.close.iloc[-1]
        row = {
            "date": as_of.isoformat(),
            "equity": self.equity(prices),
            "cash": self.cash,
            "n_positions": len(self.positions),
        }
        self.history = [h for h in self.history if h["date"] != row["date"]]
        self.history.append(row)
        self.history.sort(key=lambda h: h["date"])
        return row

    def equity_series(self) -> pd.Series:
        if not self.history:
            return pd.Series(dtype=float)
        frame = pd.DataFrame(self.history)
        return pd.Series(
            frame["equity"].to_numpy(),
            index=pd.to_datetime(frame["date"], utc=True),
            name="paper_equity",
        )


# --------------------------------------------------------------------------- #
# the comparison that matters
# --------------------------------------------------------------------------- #
def divergence_report(
    account: PaperAccount,
    panel: Panel,
    *,
    periods_per_year: float = 252.0,
) -> Dict[str, float]:
    """Compare the live account against the same strategy backtested over the
    same dates.

    Three numbers matter, in this order:

    * **tracking error** - if the live account and the backtest of the identical
      strategy over the identical dates disagree, something is wrong with the
      execution assumptions, not with the market.
    * **return correlation** - the paper account should move day-to-day with its
      own backtest. Low correlation with a similar total return means you got
      lucky, not that the model works.
    * **live vs expected Sharpe** - the weakest of the three, because a few
      months of live data cannot settle a Sharpe ratio. Reported last on purpose.
    """
    from .cross_section import SingleFeatureRanker as _Ranker
    from .engine import BacktestEngine, ExecutionConfig
    from .metrics import summarize
    from .risk import RiskConfig

    live = account.equity_series()
    if len(live) < 3:
        return {"bars": float(len(live))}

    start, end = live.index[0], live.index[-1]
    rules = PortfolioRules(
        long_only=True, top_frac=account.top_frac, gross=1.0,
        max_weight=account.max_weight, rebalance_every=21,
    )
    feats = feature_panel(panel, periods_per_year=periods_per_year)
    weights = scores_to_weights(_Ranker(account.strategy).score(feats), rules)
    frames = {k: v.loc[start:end] for k, v in panel.to_frames().items()}
    engine = BacktestEngine(
        ExecutionConfig(initial_capital=float(live.iloc[0]), periods_per_year=periods_per_year,
                        fee_bps=5.0, slippage_bps=3.0, max_leverage=1.0, min_trade_frac=0.10),
        RiskConfig(target_vol=0.0, atr_stop_mult=0.0, max_drawdown_stop=0.0,
                   reentry_lockout_bars=0),
    )
    backtest = engine.run(frames, weights.loc[start:end]).equity

    aligned = pd.DataFrame({"live": live, "backtest": backtest}).dropna()
    if len(aligned) < 3:
        return {"bars": float(len(aligned))}

    live_ret = aligned["live"].pct_change().dropna()
    bt_ret = aligned["backtest"].pct_change().dropna()
    diff = live_ret - bt_ret
    stats = summarize(
        type("R", (), {
            "equity": aligned["live"], "returns": live_ret.reindex(aligned.index).fillna(0.0),
            "weights": pd.DataFrame(1.0, index=aligned.index, columns=["book"]),
            "trades": pd.DataFrame(), "costs": pd.DataFrame(index=aligned.index),
            "exec_config": ExecutionConfig(initial_capital=float(live.iloc[0]),
                                           periods_per_year=periods_per_year),
            "meta": {},
        })()
    )
    return {
        "bars": float(len(aligned)),
        "days_live": float((end - start).days),
        "live_equity": float(aligned["live"].iloc[-1]),
        "backtest_equity": float(aligned["backtest"].iloc[-1]),
        "live_return": float(aligned["live"].iloc[-1] / aligned["live"].iloc[0] - 1.0),
        "backtest_return": float(aligned["backtest"].iloc[-1] / aligned["backtest"].iloc[0] - 1.0),
        "tracking_error_annual": float(diff.std(ddof=0) * np.sqrt(periods_per_year)),
        "return_correlation": float(live_ret.corr(bt_ret)) if len(live_ret) > 2 else float("nan"),
        "live_sharpe": float(stats["sharpe"]),
        "live_max_drawdown": float(stats["max_drawdown"]),
        "expected_annual_return": float(account.expected_annual_return),
    }


def format_divergence(report: Dict[str, float]) -> str:
    if report.get("bars", 0) < 3:
        return "  not enough live history yet - mark the account on a few more days"

    def verdict(report: Dict[str, float]) -> str:
        corr = report.get("return_correlation", float("nan"))
        te = report.get("tracking_error_annual", float("nan"))
        if not np.isfinite(corr):
            return "too early to say"
        if corr > 0.9 and te < 0.05:
            return "live account is tracking its backtest closely - the model is being executed as designed"
        if corr > 0.7:
            return "tracking loosely - check fill prices and rebalance timing before blaming the market"
        return ("live account is NOT tracking its backtest - the execution assumptions are wrong "
                "somewhere, and the backtest cannot be trusted until they agree")

    return "\n".join([
        "== paper vs backtest ==",
        f"  live for          {int(report['days_live'])} days ({int(report['bars'])} marks)",
        f"  live equity       ${report['live_equity']:,.2f}  ({report['live_return']:+.1%})",
        f"  backtest says     ${report['backtest_equity']:,.2f}  ({report['backtest_return']:+.1%})",
        f"  tracking error    {report['tracking_error_annual']:.1%} annualised",
        f"  return corr       {report['return_correlation']:.2f}",
        f"  live sharpe       {report['live_sharpe']:.2f}   max drawdown {report['live_max_drawdown']:.1%}",
        f"  verdict           {verdict(report)}",
    ])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_universe_panel(account: PaperAccount, start: str = "2016-01-01") -> Panel:
    names = UNIVERSES.get(account.universe, account.universe)
    return load_panel(names, start=start, source=account.source, min_bars=400, verbose=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradingagent.paper", description="Paper-trade the agent.")
    p.add_argument("command", choices=["init", "rebalance", "mark", "report", "positions"])
    p.add_argument("--state", default=DEFAULT_STATE)
    p.add_argument("--capital", type=float, default=100.0)
    p.add_argument("--universe", default="us_large_cap")
    p.add_argument("--source", default="yahoo")
    p.add_argument("--strategy", default="mom_12_1")
    p.add_argument("--top-frac", type=float, default=0.10)
    p.add_argument("--dry-run", action="store_true", help="show orders without recording them")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "init":
        account = PaperAccount(
            capital=args.capital, cash=args.capital, universe=args.universe,
            source=args.source, strategy=args.strategy, top_frac=args.top_frac,
            started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        account.save(args.state)
        print(f"opened a paper account with ${args.capital:,.2f} on '{args.universe}'")
        print(f"state: {args.state}")
        print("next: python -m tradingagent.paper rebalance")
        return 0

    if not os.path.exists(args.state):
        print(f"no paper account at {args.state} - run 'init' first", file=sys.stderr)
        return 1
    account = PaperAccount.load(args.state)

    if args.command == "positions":
        panel = _load_universe_panel(account)
        prices = panel.close.iloc[-1]
        print(f"cash ${account.cash:,.2f}   equity ${account.equity(prices):,.2f}")
        if not account.positions:
            print("  no open positions")
        for sym, pos in sorted(account.positions.items()):
            price = float(prices.get(sym, pos.avg_price))
            pnl = (price / pos.avg_price - 1.0) if pos.avg_price else 0.0
            print(f"  {sym:<6} {pos.shares:>10.4f} sh  @ ${pos.avg_price:>9,.2f}  "
                  f"now ${price:>9,.2f}  {pnl:+7.1%}")
        return 0

    panel = _load_universe_panel(account)

    if args.command == "rebalance":
        orders = account.plan_orders(panel)
        if orders.empty:
            print("no orders - the book already matches the target basket")
        else:
            print(f"orders as of {panel.index[-1].date()} (fill at the next open):\n")
            print(orders.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
            print(f"\n  total traded: ${orders['notional'].sum():,.2f}")
        if args.dry_run:
            print("\n(dry run - nothing recorded)")
            return 0
        account.apply_orders(orders, panel.close.iloc[-1], panel.index[-1])
        account.mark(panel)
        account.save(args.state)
        print(f"\nrecorded. equity ${account.equity(panel.close.iloc[-1]):,.2f}")
        return 0

    if args.command == "mark":
        row = account.mark(panel)
        account.save(args.state)
        print(f"{row['date'][:10]}  equity ${row['equity']:,.2f}  "
              f"cash ${row['cash']:,.2f}  {row['n_positions']} positions")
        return 0

    if args.command == "report":
        print(format_divergence(divergence_report(account, panel)))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
