"""Backtest engine: execution, costs, leverage, stops and compounding.

The engine is a bar-by-bar loop rather than a vectorised dot product, because
three things it must model are path dependent: equity compounding, intrabar
stop-losses and the drawdown kill switch.

Timing convention
-----------------
``target_weight[t]`` is a decision made with information up to the close of bar
``t``. The engine shifts it forward and fills it at the **open of bar t + 1**,
so no fill ever uses a price the decision could not have seen.

Cost model
----------
* ``fee_bps``      - exchange fee, charged on traded notional, per side.
* ``slippage_bps`` - adverse price move on the fill, per side.
* ``borrow_rate``  - annualised financing on the leveraged part of the book.
* ``short_rate``   - annualised borrow on short notional.

All of these are charged in cash, so they compound against the account exactly
as they would in reality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import indicators as ind
from .execution import BASE_COST, CostModel, ExecutionModel, plan_rebalance
from .risk import RiskConfig


@dataclass
class ExecutionConfig:
    """Everything about how orders turn into money."""

    initial_capital: float = 100.0
    fee_bps: float = 10.0            # 0.10% per side - Coinbase Advanced taker tier
    slippage_bps: float = 5.0        # 0.05% per side
    borrow_rate: float = 0.08        # annual, on (gross exposure - equity)
    short_rate: float = 0.10         # annual, on short notional
    min_trade_frac: float = 0.10     # ignore rebalances smaller than 10% of equity;
                                     # on a $100 account, dust trades are all cost
    max_leverage: float = 2.0        # hard cap on gross exposure / equity
    periods_per_year: float = 365.0
    target_equity: float = 1000.0    # the goal this project is built around
    stop_at_target: bool = False     # if True, stop trading once the goal is hit
    ruin_equity: float = 1.0         # below this the account is treated as dead
    record_trades: bool = True       # set False in a parameter search: building a
                                     # row per fill dominates the runtime when a
                                     # hundred names are rebalanced thousands of
                                     # times, and the search only reads summary
                                     # statistics
    #: The canonical cost model. When None one is derived from ``fee_bps`` and
    #: ``slippage_bps`` so existing callers keep their exact assumptions; pass a
    #: :class:`~tradingagent.execution.CostModel` to opt into spread and impact
    #: modelling, or use ``execution.cost_scenario("low"|"base"|"high")``.
    costs: Optional[CostModel] = None
    #: Where a fill happens relative to the bar that produced the signal.
    fill_at: str = "next_open"

    def cost_model(self) -> CostModel:
        """The cost model this config implies.

        Derived from the legacy bps fields when none was supplied, which keeps
        every historical result reproducible: the spread component is folded
        into slippage rather than added on top.
        """
        if self.costs is not None:
            return self.costs
        return CostModel(
            fee_bps=self.fee_bps,
            half_spread_bps=0.0,
            slippage_bps=self.slippage_bps,
            impact_bps_at_full=0.0,
            borrow_rate=self.borrow_rate,
            short_rate=self.short_rate,
            name="derived",
        )

    def execution_model(self) -> ExecutionModel:
        return ExecutionModel(
            fill_at=self.fill_at,
            costs=self.cost_model(),
            min_trade_frac=self.min_trade_frac,
        )


@dataclass
class BacktestResult:
    """Everything a run produced, ready for metrics and plots."""

    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame           # realised weight per asset, per bar
    target_weights: pd.DataFrame    # what was asked for, after the risk layer
    trades: pd.DataFrame
    costs: pd.DataFrame             # fees, slippage and financing per bar
    exec_config: ExecutionConfig
    risk_config: RiskConfig
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else float("nan")

    @property
    def hit_target(self) -> bool:
        return bool((self.equity >= self.exec_config.target_equity).any())

    def stats(self, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
        from .metrics import summarize

        return summarize(self, benchmark=benchmark)


class BacktestEngine:
    """Run one or many assets against a shared, compounding pot of capital."""

    def __init__(self, exec_config: ExecutionConfig | None = None, risk_config: RiskConfig | None = None):
        self.exec = exec_config or ExecutionConfig()
        self.risk = risk_config or RiskConfig()

    # ------------------------------------------------------------------ #
    def run(
        self,
        data: pd.DataFrame | Dict[str, pd.DataFrame],
        target_weights: pd.Series | pd.DataFrame,
    ) -> BacktestResult:
        """Backtest ``target_weights`` (decision at bar ``t``) over ``data``."""
        panel = self._as_panel(data)
        symbols = list(panel.keys())
        index = panel[symbols[0]].index

        tw = self._as_weight_frame(target_weights, symbols, index)
        # the decision at bar t is filled at the open of bar t + 1
        tw_exec = tw.shift(1).fillna(0.0)

        ec, rc = self.exec, self.risk
        exec_model = ec.execution_model()

        opens = np.column_stack([panel[s]["open"].to_numpy() for s in symbols])
        highs = np.column_stack([panel[s]["high"].to_numpy() for s in symbols])
        lows = np.column_stack([panel[s]["low"].to_numpy() for s in symbols])
        closes = np.column_stack([panel[s]["close"].to_numpy() for s in symbols])
        volumes = np.column_stack([panel[s]["volume"].to_numpy() for s in symbols])

        # A symbol that has not listed yet - or has stopped trading - is NaN.
        # Tradeability is decided on the OPEN alone, because that is all you
        # know when the order goes in: whether this bar will produce a closing
        # print is not knowable at the open, and requiring it would let the
        # backtest skip a name on its final day using information from the end
        # of that day.
        # the price a scheduled order actually executes against
        fills = opens if ec.fill_at == "next_open" else closes
        tradeable = np.isfinite(fills) & (fills > 0)
        # Marking falls back to the open, then to the last price that printed,
        # so the arithmetic stays finite through a half-formed bar.
        mark_df = pd.DataFrame(closes).where(np.isfinite(closes) & (closes > 0), pd.DataFrame(opens))
        mark = mark_df.ffill().to_numpy()
        mark = np.where(np.isfinite(mark), mark, 0.0)
        # ATR is only needed when a protective stop is armed. Computing it
        # unconditionally cost a full pass over every symbol on every call, and
        # a walk-forward makes thousands of calls over overlapping windows with
        # stops switched off.
        if rc.atr_stop_mult > 0 or rc.take_profit_mult > 0:
            atrs = np.column_stack(
                [
                    ind.atr(panel[s]["high"], panel[s]["low"], panel[s]["close"], rc.atr_n)
                    .shift(1)  # the ATR that was known when the order was placed
                    .to_numpy()
                    for s in symbols
                ]
            )
        else:
            atrs = np.full((len(index), len(symbols)), np.nan)
        w_des = tw_exec.to_numpy()

        n_bars, n_assets = closes.shape
        cost_rate = exec_model.costs.rate()
        impact_on = exec_model.costs.impact_bps_at_full > 0

        cash = float(ec.initial_capital)
        units = np.zeros(n_assets)
        stop_px = np.zeros(n_assets)
        tp_px = np.zeros(n_assets)
        lockout = np.zeros(n_assets, dtype=int)   # bars left to stand aside per asset
        entry_px = np.zeros(n_assets)             # price a position was opened at

        equity_arr = np.empty(n_bars)
        weight_arr = np.zeros((n_bars, n_assets))
        fee_arr = np.zeros(n_bars)
        fin_arr = np.zeros(n_bars)
        trades: List[dict] = []

        equity = float(ec.initial_capital)
        peak = equity
        cooldown = 0
        kill_switch_events = 0
        n_trades = 0
        traded_notional = 0.0
        record = bool(ec.record_trades)
        stops_enabled = rc.atr_stop_mult > 0 or rc.take_profit_mult > 0
        dead = False
        frozen = False  # set when stop_at_target fires

        for i in range(n_bars):
            if dead:
                equity_arr[i] = 0.0
                continue

            equity_prev = equity
            desired = w_des[i].copy()

            # ---- capital protection ------------------------------------ #
            if frozen:
                desired[:] = 0.0
            if cooldown > 0:
                desired[:] = 0.0
                cooldown -= 1
                if cooldown == 0:
                    # Restart the drawdown clock from where the account actually
                    # is. Without this reset the kill switch latches on for good:
                    # equity stays below the old high-water mark, so every
                    # subsequent bar re-arms the cooldown and the agent never
                    # trades again.
                    peak = equity_prev
            elif rc.max_drawdown_stop > 0 and equity_prev <= peak * (1.0 - rc.max_drawdown_stop):
                desired[:] = 0.0
                cooldown = int(rc.cooldown_bars)
                kill_switch_events += 1

            # a freshly stopped-out asset is off limits until its lockout ends
            for j in np.nonzero(lockout)[0]:
                lockout[j] -= 1
                if units[j] == 0.0:
                    desired[j] = 0.0

            # names that cannot be traded this bar: no new exposure, and any
            # position left stranded in one is closed at its last printed price
            live = tradeable[i]
            bar_fees = 0.0
            if not live.all():
                desired = np.where(live, desired, 0.0)
            for j in np.nonzero(~live & (units != 0.0))[0]:
                exit_px = mark[i, j]
                fee = abs(units[j]) * exit_px * cost_rate
                cash += units[j] * exit_px - fee
                bar_fees += fee
                if record:
                    trades.append(
                        {
                            "timestamp": index[i],
                            "symbol": symbols[j],
                            "side": "sell" if units[j] > 0 else "buy",
                            "units": float(-units[j]),
                            "price": float(exit_px),
                            "notional": float(abs(units[j]) * exit_px),
                            "cost": float(fee),
                            "reason": "delisted",
                            "equity_before": float(equity_prev),
                        }
                    )
                n_trades += 1
                traded_notional += abs(units[j]) * exit_px
                units[j] = 0.0
                stop_px[j] = tp_px[j] = entry_px[j] = 0.0

            # ---- gross exposure cap ------------------------------------ #
            gross = float(np.abs(desired).sum())
            if gross > ec.max_leverage and gross > 0:
                desired *= ec.max_leverage / gross

            # ---- fill at the execution reference price ----------------- #
            # Sizing, dust filtering and fill pricing all come from the shared
            # planner in execution.py, so the backtest and the paper account
            # cannot drift apart. The reference is this bar's open (or close,
            # under fill_at="next_close"); the weight being filled was formed on
            # the previous bar's close.
            px_open = np.where(live, fills[i], mark[i])
            plan = plan_rebalance(
                desired,
                units,
                equity_prev,
                px_open,
                model=exec_model,
                tradeable=live,
                symbols=tuple(symbols),
                bar_volume=volumes[i] if impact_on else None,
            )
            delta = plan.delta_units
            notional = plan.notional

            for j in plan.nonzero():
                # the adverse fill price already carries fee + spread + slippage
                # + impact, so the cash flow is delta x fill_price and nothing
                # else. Components are split out only for reporting.
                fill_px = float(plan.fill_price[j])
                # the cost actually paid is the gap between the reference price
                # and the fill, times the size - which carries impact when the
                # cost model charges it
                fee = abs(delta[j] * (fill_px - px_open[j]))
                cash -= delta[j] * fill_px
                bar_fees += fee
                prev_units = units[j]
                units[j] += delta[j]
                if record:
                    trades.append(
                        {
                            "timestamp": index[i],
                            "symbol": symbols[j],
                            "side": "buy" if delta[j] > 0 else "sell",
                            "units": float(delta[j]),
                            "price": float(fill_px),
                            "reference_price": float(px_open[j]),
                            "notional": float(notional[j]),
                            "cost": float(fee),
                            "reason": "rebalance",
                            "equity_before": float(equity_prev),
                        }
                    )
                n_trades += 1
                traded_notional += notional[j]
                # (re)arm the protective stop whenever a position opens or flips
                if np.sign(units[j]) != np.sign(prev_units) or prev_units == 0.0:
                    entry_px[j] = px_open[j]
                    stop_px[j], tp_px[j] = self._init_stops(units[j], px_open[j], atrs[i, j])

            # A position opened before the ATR had enough history would other-
            # wise run unprotected for ever; arm it as soon as ATR exists.
            if stops_enabled:
                for j in np.nonzero(units)[0]:
                    if stop_px[j] == 0.0 and np.isfinite(atrs[i, j]):
                        stop_px[j], tp_px[j] = self._init_stops(units[j], entry_px[j], atrs[i, j])

            # ---- protective exits ------------------------------------- #
            # The stop level tested here was fixed before this bar opened, so a
            # fill within the bar is achievable. Trailing happens *after* the
            # test, using this bar's extremes, and only takes effect next bar -
            # trailing first would let a stop ride up on a high that had not
            # printed yet when the low came in.
            for j in (np.nonzero(units)[0] if stops_enabled else ()):
                if not live[j]:
                    continue

                exit_px = self._stop_exit_price(
                    units[j], px_open[j], highs[i, j], lows[i, j], stop_px[j], tp_px[j]
                )
                if exit_px is not None:
                    qty = -units[j]                      # the closing trade
                    fee = abs(qty) * exit_px * cost_rate
                    cash += units[j] * exit_px - fee      # sell longs / buy back shorts
                    bar_fees += fee
                    if record:
                        trades.append(
                            {
                                "timestamp": index[i],
                                "symbol": symbols[j],
                                "side": "sell" if units[j] > 0 else "buy",
                                "units": float(qty),
                                "price": float(exit_px),
                                "notional": float(abs(qty) * exit_px),
                                "cost": float(fee),
                                "reason": "stop",
                                "equity_before": float(equity_prev),
                            }
                        )
                    n_trades += 1
                    traded_notional += abs(qty) * exit_px
                    units[j] = 0.0
                    stop_px[j] = tp_px[j] = entry_px[j] = 0.0
                    # having just been stopped out, stand aside for a while
                    # instead of buying the same signal straight back
                    lockout[j] = int(rc.reentry_lockout_bars)
                    continue

                if rc.trail_stop and rc.atr_stop_mult > 0 and np.isfinite(atrs[i, j]):
                    band = rc.atr_stop_mult * atrs[i, j]
                    if units[j] > 0:
                        stop_px[j] = max(stop_px[j], highs[i, j] - band)
                    elif stop_px[j] > 0:
                        stop_px[j] = min(stop_px[j], lows[i, j] + band)
                    else:
                        stop_px[j] = lows[i, j] + band

            # ---- financing on leverage and shorts ---------------------- #
            px_close = mark[i]
            exposure = units * px_close
            gross_exposure = float(np.abs(exposure).sum())
            short_notional = float(np.abs(np.minimum(exposure, 0.0)).sum())
            mtm_equity = cash + float(exposure.sum())
            financing = exec_model.costs.financing_per_bar(
                gross_exposure, mtm_equity, short_notional, ec.periods_per_year
            )
            cash -= financing

            equity = cash + float(exposure.sum())
            equity_arr[i] = equity
            weight_arr[i] = exposure / equity if equity > 0 else 0.0
            fee_arr[i] = bar_fees
            fin_arr[i] = financing

            if equity > peak:
                peak = equity

            # ---- ruin and goal ----------------------------------------- #
            if equity <= ec.ruin_equity:
                dead = True
                equity_arr[i] = max(equity, 0.0)
                cash, units[:] = 0.0, 0.0
                equity = 0.0
                continue
            if ec.stop_at_target and equity >= ec.target_equity:
                frozen = True

        equity_s = pd.Series(equity_arr, index=index, name="equity")
        returns = equity_s.pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df = trades_df.set_index("timestamp")

        return BacktestResult(
            equity=equity_s,
            returns=returns,
            weights=pd.DataFrame(weight_arr, index=index, columns=symbols),
            target_weights=tw_exec,
            trades=trades_df,
            costs=pd.DataFrame({"fees": fee_arr, "financing": fin_arr}, index=index),
            exec_config=ec,
            risk_config=rc,
            meta={
                "symbols": symbols,
                "bust": bool(dead),
                "kill_switch_events": int(kill_switch_events),
                "n_trades": int(n_trades),
                "traded_notional": float(traded_notional),
            },
        )

    # ------------------------------------------------------------------ #
    def _init_stops(self, units: float, price: float, atr_val: float):
        rc = self.risk
        if units == 0.0 or not np.isfinite(atr_val) or rc.atr_stop_mult <= 0:
            return 0.0, 0.0
        dist = rc.atr_stop_mult * atr_val
        if units > 0:
            stop = max(price - dist, 0.0)
            tp = price + rc.take_profit_mult * atr_val if rc.take_profit_mult > 0 else 0.0
        else:
            stop = price + dist
            tp = max(price - rc.take_profit_mult * atr_val, 0.0) if rc.take_profit_mult > 0 else 0.0
        return stop, tp

    @staticmethod
    def _stop_exit_price(units, open_px, high, low, stop, tp) -> Optional[float]:
        """Price at which a protective order would have filled this bar, if any.

        When the bar gaps through the stop the fill happens at the open, not at
        the stop price - the pessimistic assumption, and the realistic one.
        """
        if units == 0.0:
            return None
        if units > 0:
            if stop > 0 and low <= stop:
                return float(min(stop, open_px))   # a gap-down fills at the open
            if tp > 0 and high >= tp:
                return float(max(tp, open_px))
        else:
            if stop > 0 and high >= stop:
                return float(max(stop, open_px))   # a gap-up fills at the open
            if tp > 0 and low <= tp:
                return float(min(tp, open_px))
        return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_panel(data) -> Dict[str, pd.DataFrame]:
        if isinstance(data, pd.DataFrame):
            return {"asset": data}
        panel = dict(data)
        idx = None
        for df in panel.values():
            if idx is None:
                idx = df.index
            elif not df.index.equals(idx):
                raise ValueError(
                    "all symbols must share one index; use Panel.from_frames or "
                    "data.align_universe"
                )
        return panel

    @staticmethod
    def _as_weight_frame(weights, symbols: Sequence[str], index: pd.Index) -> pd.DataFrame:
        if isinstance(weights, pd.Series):
            if len(symbols) != 1:
                raise ValueError("a Series of weights needs exactly one symbol")
            frame = weights.to_frame(symbols[0])
        else:
            frame = weights.copy()
            missing = [s for s in symbols if s not in frame.columns]
            if missing:
                raise ValueError(f"weights missing columns for {missing}")
            frame = frame[list(symbols)]
        return frame.reindex(index).fillna(0.0).astype(float)


def buy_and_hold_equity(
    df: pd.DataFrame,
    initial_capital: float = 100.0,
    fee_bps: Optional[float] = None,
    *,
    costs: Optional[CostModel] = None,
) -> pd.Series:
    """Benchmark curve: buy at the first open, hold, pay one entry cost.

    Charged the same way a strategy is charged - one crossing of the full cost
    model on entry. ``fee_bps`` is kept for backwards compatibility; when both
    are omitted the BASE scenario applies.
    """
    model = costs or (
        CostModel(fee_bps=fee_bps, half_spread_bps=0.0, slippage_bps=0.0)
        if fee_bps is not None
        else BASE_COST
    )
    px = df["close"]
    entry = float(model.fill_price(df["open"].iloc[0], +1))
    units = initial_capital / entry
    return (units * px).rename("buy_and_hold")
