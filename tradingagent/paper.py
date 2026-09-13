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
from .execution import DEFAULT_EXECUTION, ExecutionModel, plan_rebalance
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
    #: Orders decided on a bar's close and waiting for the next execution
    #: opportunity. They are NOT positions yet. Persisting them is what makes
    #: paper trading obey the same one-bar lag the backtest does: you cannot
    #: decide and fill in the same instant.
    pending: List[dict] = field(default_factory=list)
    fill_at: str = "next_open"
    universe: str = "us_large_cap"
    source: str = "yahoo"
    strategy: str = "mom_12_1"
    top_frac: float = 0.10
    max_weight: float = 0.15
    started: str = ""
    expected_annual_return: float = 0.0    # what the backtest said, for comparison
    expected_annual_vol: float = 0.0
    #: Cost scenario name; resolved against execution.COST_SCENARIOS.
    cost_scenario: str = "base"
    #: Bars per year, used to accrue financing at the same rate the engine does.
    periods_per_year: float = 252.0
    #: Index position of the last bar financing was charged for, so an
    #: irregularly-marked account still accrues one bar's cost per bar.
    last_financed_bar: Optional[str] = None

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

    def execution_model(self, min_trade_frac: float = 0.10) -> ExecutionModel:
        """The same execution model the backtest uses, from the same scenario."""
        from .execution import cost_scenario

        return ExecutionModel(
            fill_at=self.fill_at,
            costs=cost_scenario(self.cost_scenario),
            min_trade_frac=min_trade_frac,
        )

    def plan_orders(
        self, panel: Panel, *, min_trade_frac: float = 0.10,
        target: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Orders that would move the book to its target basket.

        Decided on the **last closed bar**, and deliberately NOT executed here:
        the prices used for sizing are the execution reference the next bar will
        provide, which does not exist yet. :meth:`schedule_orders` records them
        as pending; :meth:`fill_pending` executes them once that bar arrives.

        Sizing therefore uses the last close as an *estimate* of the execution
        price, and the fill re-sizes against the price that actually printed -
        exactly what a notional (fractional-share) order does.
        """
        prices = panel.close.iloc[-1]
        equity = self.equity(prices)
        if target is None:
            target = self.target_basket(panel)

        symbols = sorted(set(target.index) | set(self.positions))
        current = np.array([
            self.positions[s].shares if s in self.positions else 0.0 for s in symbols
        ])
        reference = np.array([float(prices.get(s, np.nan)) for s in symbols])
        weights = np.array([float(target.get(s, 0.0)) for s in symbols])

        plan = plan_rebalance(
            weights, current, equity, reference,
            model=self.execution_model(min_trade_frac),
            symbols=tuple(symbols),
        )
        rows = []
        for j in plan.nonzero():
            delta = float(plan.delta_units[j])
            rows.append({
                "symbol": symbols[j],
                "side": "BUY" if delta > 0 else "SELL",
                "shares": abs(delta),
                "target_weight": float(weights[j]),
                "estimated_price": float(reference[j]),
                "estimated_notional": float(plan.notional[j]),
                "decided_on": panel.index[-1].isoformat(),
            })
        return pd.DataFrame(rows)

    def schedule_orders(self, orders: pd.DataFrame, decided_on: pd.Timestamp) -> int:
        """Queue orders for the next execution opportunity.

        Replaces any orders still pending: a basket decided today supersedes one
        decided yesterday that never filled.
        """
        self.pending = []
        if orders is None or orders.empty:
            return 0
        for _, row in orders.iterrows():
            self.pending.append({
                "symbol": row["symbol"],
                "side": row["side"],
                "target_weight": float(row.get("target_weight", 0.0)),
                "estimated_price": float(row.get("estimated_price", np.nan)),
                "decided_on": pd.Timestamp(decided_on).isoformat(),
            })
        return len(self.pending)

    def fill_pending(self, panel: Panel, *, min_trade_frac: float = 0.10) -> pd.DataFrame:
        """Execute queued orders against the current bar's execution price.

        Refuses to fill on the bar the orders were decided on. That single check
        is what keeps paper trading causally honest, and
        ``tests/test_execution_parity.py`` fails loudly if it is ever removed.
        """
        if not self.pending:
            return pd.DataFrame()

        bar = panel.index[-1]
        decided = pd.Timestamp(self.pending[0]["decided_on"])
        if bar <= decided:
            raise ValueError(
                f"orders were decided on {decided.date()} and cannot fill on {bar.date()}; "
                "a signal formed at a bar's close executes at the NEXT bar's "
                "execution price, never that same bar"
            )

        model = self.execution_model(min_trade_frac)
        column = model.fill_column()
        reference_row = getattr(panel, column).iloc[-1]
        # Size against equity as of the last close BEFORE this bar. Using this
        # bar's close would be a look-ahead: it prints after the open the order
        # fills at. The backtest sizes the same way (its `equity_prev`).
        prior_close = panel.close.iloc[-2] if len(panel) > 1 else panel.close.iloc[-1]
        equity = self.equity(prior_close)

        symbols = [o["symbol"] for o in self.pending]
        weights = np.array([o["target_weight"] for o in self.pending])
        current = np.array([
            self.positions[s].shares if s in self.positions else 0.0 for s in symbols
        ])
        reference = np.array([float(reference_row.get(s, np.nan)) for s in symbols])

        volume = np.array([float(panel.volume.iloc[-1].get(s, np.nan)) for s in symbols])
        plan = plan_rebalance(
            weights, current, equity, reference,
            model=model, symbols=tuple(symbols), bar_volume=volume,
        )
        filled = self._execute_plan(plan, symbols, bar, model)
        self.pending = []
        return filled

    def _execute_plan(self, plan, symbols, as_of, model) -> pd.DataFrame:
        """Book the fills from a plan. The only place paper positions change."""
        rows = []
        for j in plan.nonzero():
            sym = symbols[j]
            shares = float(plan.delta_units[j])
            fill_px = float(plan.fill_price[j])
            reference = float(plan.reference_price[j])
            cost = abs(shares * (fill_px - reference))
            components = model.costs.split(plan.notional[j])
            # rescale the split so the parts sum to what was actually paid
            total_split = sum(components.values())
            if total_split > 0:
                components = {k: v * cost / total_split for k, v in components.items()}
            # the adverse fill price already carries every cost component
            self.cash -= shares * fill_px
            self._book(sym, shares, fill_px)
            record = {
                "at": pd.Timestamp(as_of).isoformat(),
                "symbol": sym,
                "side": "BUY" if shares > 0 else "SELL",
                "shares": abs(shares),
                "price": fill_px,
                "reference_price": reference,
                "notional": float(plan.notional[j]),
                "cost": cost,
                **components,
            }
            self.orders.append(record)
            rows.append(record)
        return pd.DataFrame(rows)

    def _book(self, symbol: str, shares: float, price: float) -> None:
        """Update one position, keeping the average price honest on both sides."""
        existing = self.positions.get(symbol)
        prior = existing.shares if existing else 0.0
        new_shares = prior + shares
        if abs(new_shares) < 1e-9:
            self.positions.pop(symbol, None)
            return
        if existing is None or prior == 0.0 or np.sign(new_shares) != np.sign(prior):
            # opening, or flipping through zero: the basis resets
            self.positions[symbol] = Position(symbol, new_shares, price)
        elif abs(new_shares) > abs(prior):
            # adding to a position: weighted-average the basis
            basis = (existing.avg_price * prior + price * shares) / new_shares
            self.positions[symbol] = Position(symbol, new_shares, basis)
        else:
            # reducing: the basis of what remains is unchanged
            self.positions[symbol] = Position(symbol, new_shares, existing.avg_price)

    def apply_orders(
        self, orders: pd.DataFrame, prices: pd.Series, as_of: pd.Timestamp, *, cost_bps=None
    ) -> None:
        """Deprecated: fills orders immediately at the supplied prices.

        Kept so existing callers and tests keep working, but it models something
        the backtest does not - execution at the price the decision was made on.
        Use :meth:`schedule_orders` then :meth:`fill_pending`.
        """
        import warnings as _warnings

        _warnings.warn(
            "apply_orders fills at the price supplied rather than at the next "
            "execution opportunity; use schedule_orders() + fill_pending() to "
            "match the backtest's timing",
            DeprecationWarning,
            stacklevel=2,
        )
        if orders is None or orders.empty:
            return
        model = self.execution_model()
        for _, row in orders.iterrows():
            sym = row["symbol"]
            reference = float(prices.get(sym, row.get("estimated_price", np.nan)))
            if not np.isfinite(reference) or reference <= 0:
                continue
            shares = float(row["shares"]) * (1 if row["side"] == "BUY" else -1)
            fill_px = float(model.costs.fill_price(reference, np.sign(shares)))
            notional = abs(shares) * reference
            components = model.costs.split(notional)
            self.cash -= shares * fill_px
            self._book(sym, shares, fill_px)
            self.orders.append({
                "at": pd.Timestamp(as_of).isoformat(), "symbol": sym,
                "side": row["side"], "shares": abs(shares), "price": fill_px,
                "reference_price": reference, "notional": notional,
                "cost": sum(components.values()), **components,
            })

    def accrue_financing(self, panel: Panel) -> float:
        """Charge financing on borrowed capital and borrow on shorts.

        The backtest charges this every bar. Paper must too, or the two drift:
        paying an adverse fill price on a fully-invested book leaves cash
        slightly negative, which is borrowed money, and ignoring it made the
        paper account quietly cheaper than its own backtest.

        Accrues one bar's cost per elapsed bar, so an account marked weekly is
        charged for the whole week rather than for a single day.
        """
        as_of = panel.index[-1]
        model = self.execution_model()
        prices = panel.close.iloc[-1]
        exposure = sum(
            pos.shares * float(prices.get(sym, pos.avg_price))
            for sym, pos in self.positions.items()
        )
        gross = sum(
            abs(pos.shares * float(prices.get(sym, pos.avg_price)))
            for sym, pos in self.positions.items()
        )
        short_notional = sum(
            abs(pos.shares * float(prices.get(sym, pos.avg_price)))
            for sym, pos in self.positions.items() if pos.shares < 0
        )
        equity = self.cash + exposure

        bars = 1
        if self.last_financed_bar is not None:
            previous = pd.Timestamp(self.last_financed_bar)
            if as_of <= previous:
                return 0.0
            position = panel.index.searchsorted(previous)
            bars = max(int(len(panel.index) - 1 - position), 1)

        charge = bars * model.costs.financing_per_bar(
            gross, equity, short_notional, self.periods_per_year
        )
        self.cash -= charge
        self.last_financed_bar = as_of.isoformat()
        return charge

    def mark(self, panel: Panel) -> dict:
        """Record today's equity, after accruing financing. Idempotent per date."""
        self.accrue_financing(panel)
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
    periods_per_year: Optional[float] = None,
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
    from .execution import cost_scenario
    from .metrics import summarize
    from .risk import RiskConfig

    # The calendar must come from the account being judged. A default that
    # disagrees with it charges financing at a different rate in the comparison
    # backtest than the account itself paid, and the gap shows up as tracking
    # error that is really the framework arguing with itself.
    if periods_per_year is None:
        periods_per_year = account.periods_per_year

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
                        costs=cost_scenario(account.cost_scenario), max_leverage=1.0,
                        min_trade_frac=0.10),
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

    # Annualise by how often the account is actually marked, not by assuming
    # daily. Marking weekly and scaling by sqrt(252) inflates tracking error by
    # sqrt(5) and turns a well-behaved account into an alarming one.
    spacing = np.median(np.diff(aligned.index.values).astype("timedelta64[D]").astype(float))
    marks_per_year = periods_per_year / max(spacing * periods_per_year / 365.0, 1.0)
    stats = summarize(
        type("R", (), {
            "equity": aligned["live"], "returns": live_ret.reindex(aligned.index).fillna(0.0),
            "weights": pd.DataFrame(1.0, index=aligned.index, columns=["book"]),
            "trades": pd.DataFrame(), "costs": pd.DataFrame(index=aligned.index),
            "exec_config": ExecutionConfig(initial_capital=float(live.iloc[0]),
                                           periods_per_year=marks_per_year),
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
        "tracking_error_annual": float(diff.std(ddof=0) * np.sqrt(marks_per_year)),
        "marks_per_year": float(marks_per_year),
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
        f"  tracking error    {report['tracking_error_annual']:.1%} annualised "
        f"(from {report.get('marks_per_year', float('nan')):.0f} marks/yr)",
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
        # first: anything decided on a previous run fills against today's bar
        filled = account.fill_pending(panel) if account.pending else pd.DataFrame()
        if not filled.empty:
            print(f"filled {len(filled)} order(s) queued earlier, at "
                  f"{panel.index[-1].date()} {account.fill_at.replace('next_', '')}:\n")
            print(filled[["symbol", "side", "shares", "price", "notional", "cost"]]
                  .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
            print()

        orders = account.plan_orders(panel)
        if orders.empty:
            print("no new orders - the book already matches the target basket")
        else:
            print(f"orders decided on {panel.index[-1].date()}'s close, to fill at the "
                  f"NEXT bar's {account.fill_at.replace('next_', '')}:\n")
            print(orders.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
            print(f"\n  estimated notional: ${orders['estimated_notional'].sum():,.2f}")
            print("  (estimated at the last close; the fill re-sizes against the "
                  "price that actually prints)")
        if args.dry_run:
            print("\n(dry run - nothing recorded)")
            return 0
        n = account.schedule_orders(orders, panel.index[-1])
        account.mark(panel)
        account.save(args.state)
        print(f"\n{n} order(s) queued. Run 'rebalance' again on the next bar to fill them.")
        print(f"equity ${account.equity(panel.close.iloc[-1]):,.2f}")
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
