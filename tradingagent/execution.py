"""The canonical execution and cost model.

Before this module existed the project had two execution implementations — one
in :mod:`tradingagent.engine` for backtests and one in :mod:`tradingagent.paper`
for forward trading — and they disagreed about when an order fills. The backtest
filled at the next bar's open; paper filled at the *same close that generated
the signal*. Any tracking error between them was therefore partly the framework
disagreeing with itself.

Everything about turning a decision into a position now lives here, and both
paths call it.

The causal sequence
-------------------
::

    bar T closes
      └─ strategy sees information available at T's close
           └─ target weights are produced
                └─ orders are scheduled
                     └─ bar T+1 opens
                          └─ orders fill at T+1's execution price
                               └─ the book is marked at T+1

A weight formed on bar ``T``'s close can never transact at that close. The lag
is one bar, always, in both paths — :attr:`ExecutionModel.decision_lag_bars`.

The cost model
--------------
Four components, charged as an **adverse fill price** rather than a rebate from
cash, so the recorded trade price is what was actually paid:

``fee_bps``           commission / exchange fee, per side
``half_spread_bps``   half the bid/ask, crossed on entry and again on exit
``slippage_bps``      residual adverse move beyond the quoted spread
``impact_bps``        market impact, growing with participation in the bar

Financing (leverage) and borrow (shorts) are charged separately, per bar, on
the marked book.

Charging the spread as a price rather than a cash fee is *arithmetically
identical* for cash flow — ``delta × price × (1 + sign × rate)`` expands to
``delta × price + |delta| × price × rate`` — so this refactor changed no
backtest result. What it fixes is the trade log, which previously recorded the
untouched reference price and understated what a fill cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# costs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostModel:
    """What it costs to trade, in basis points of notional, per side.

    Defaults are the BASE scenario: deliberately not optimistic. They reproduce
    the framework's historical assumption of ``fee_bps=10`` plus
    ``slippage_bps=5`` exactly (10 + 2.5 + 2.5 = 15 bps per side), so adopting
    this model changed no published number.
    """

    fee_bps: float = 10.0
    half_spread_bps: float = 2.5
    slippage_bps: float = 2.5
    #: Impact in bps at 100% participation, scaled by ``participation ** exponent``.
    #: Zero by default because at the account sizes this framework targets ($100
    #: to a few thousand) participation in a liquid name is indistinguishable
    #: from zero. Turn it on before believing any capacity claim.
    impact_bps_at_full: float = 0.0
    impact_exponent: float = 0.5        # square-root law
    borrow_rate: float = 0.08           # annualised, on leverage above 1x
    short_rate: float = 0.10            # annualised, on short notional
    name: str = "base"

    # -- rates ---------------------------------------------------------- #
    def impact_bps(self, participation: float | np.ndarray = 0.0):
        """Impact in bps for a given share of the bar's volume."""
        if self.impact_bps_at_full <= 0:
            return np.zeros_like(np.asarray(participation, dtype=float))
        share = np.clip(np.asarray(participation, dtype=float), 0.0, 1.0)
        return self.impact_bps_at_full * share**self.impact_exponent

    def total_bps(self, participation: float | np.ndarray = 0.0):
        """All per-side costs combined, in basis points."""
        return self.fee_bps + self.half_spread_bps + self.slippage_bps + self.impact_bps(
            participation
        )

    def rate(self, participation: float | np.ndarray = 0.0):
        """All per-side costs combined, as a fraction."""
        return self.total_bps(participation) / 1e4

    # -- prices --------------------------------------------------------- #
    def fill_price(self, reference_price, side_sign, participation=0.0):
        """The price actually paid: reference, moved against you.

        ``side_sign`` is +1 to buy and -1 to sell. Buying lifts the offer and
        pays impact; selling hits the bid and pays it too.
        """
        return np.asarray(reference_price, dtype=float) * (
            1.0 + np.asarray(side_sign, dtype=float) * self.rate(participation)
        )

    def split(self, notional, participation=0.0) -> Dict[str, float]:
        """Break a trade's cost into its components, for reporting."""
        notional = float(np.abs(notional))
        return {
            "fee": notional * self.fee_bps / 1e4,
            "spread": notional * self.half_spread_bps / 1e4,
            "slippage": notional * self.slippage_bps / 1e4,
            "impact": notional * float(np.asarray(self.impact_bps(participation))) / 1e4,
        }

    def financing_per_bar(
        self, gross_exposure: float, equity: float, short_notional: float, periods_per_year: float
    ) -> float:
        """Financing on borrowed capital plus borrow on short notional, one bar."""
        borrowed = max(gross_exposure - max(equity, 0.0), 0.0)
        return (
            borrowed * self.borrow_rate / periods_per_year
            + short_notional * self.short_rate / periods_per_year
        )


#: Cost scenarios. BASE is the default and matches the framework's historical
#: assumption; LOW is a generous retail broker; HIGH is what a thin name or a
#: bad day actually costs. A strategy that only works under LOW is fragile, and
#: the robustness battery reports exactly that.
LOW_COST = CostModel(
    fee_bps=2.0, half_spread_bps=1.0, slippage_bps=1.0, borrow_rate=0.05,
    short_rate=0.06, name="low",
)
BASE_COST = CostModel(name="base")
HIGH_COST = CostModel(
    fee_bps=20.0, half_spread_bps=7.5, slippage_bps=7.5, impact_bps_at_full=50.0,
    borrow_rate=0.12, short_rate=0.15, name="high",
)
COST_SCENARIOS: Dict[str, CostModel] = {
    "low": LOW_COST,
    "base": BASE_COST,
    "high": HIGH_COST,
}


def cost_scenario(name: str) -> CostModel:
    if name not in COST_SCENARIOS:
        raise KeyError(f"unknown cost scenario {name!r}; known: {sorted(COST_SCENARIOS)}")
    return COST_SCENARIOS[name]


# --------------------------------------------------------------------------- #
# execution timing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecutionModel:
    """When an order fills, and what it costs when it does."""

    #: Where the fill happens relative to the bar that produced the signal.
    #: ``"next_open"`` is the default and the only one a daily strategy can
    #: honestly claim: you see the close, you queue an order, it fills on the
    #: opening auction. ``"next_close"`` models a market-on-close order placed
    #: the following session — slower, and strictly more conservative.
    fill_at: str = "next_open"
    costs: CostModel = field(default_factory=lambda: BASE_COST)
    #: Ignore rebalances smaller than this fraction of the position being
    #: adjusted. A full exit is never dust.
    min_trade_frac: float = 0.10

    @property
    def decision_lag_bars(self) -> int:
        """Bars between the decision and the fill. One, always."""
        return 1

    def fill_column(self) -> str:
        return {"next_open": "open", "next_close": "close"}[self.fill_at]

    def __post_init__(self):
        if self.fill_at not in ("next_open", "next_close"):
            raise ValueError(
                f"fill_at must be 'next_open' or 'next_close', got {self.fill_at!r}"
            )
        if not 0.0 <= self.min_trade_frac < 1.0:
            raise ValueError(
                f"min_trade_frac must be in [0, 1); {self.min_trade_frac} would "
                "disable all rebalancing"
            )


DEFAULT_EXECUTION = ExecutionModel()


# --------------------------------------------------------------------------- #
# the one place orders are sized
# --------------------------------------------------------------------------- #
@dataclass
class OrderPlan:
    """Orders to move a book to its target, before any of them are filled."""

    symbols: Tuple[str, ...]
    delta_units: np.ndarray      # signed, in units
    reference_price: np.ndarray  # the execution reference (e.g. the next open)
    fill_price: np.ndarray       # reference moved adversely by the cost model
    notional: np.ndarray         # |delta| x reference
    is_full_exit: np.ndarray
    suppressed: np.ndarray       # dust that was filtered out

    def nonzero(self) -> np.ndarray:
        return np.nonzero(self.delta_units)[0]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": list(self.symbols),
                "delta_units": self.delta_units,
                "reference_price": self.reference_price,
                "fill_price": self.fill_price,
                "notional": self.notional,
                "full_exit": self.is_full_exit,
                "suppressed": self.suppressed,
            }
        )


def plan_rebalance(
    target_weights: np.ndarray,
    current_units: np.ndarray,
    equity: float,
    reference_price: np.ndarray,
    *,
    model: ExecutionModel,
    tradeable: Optional[np.ndarray] = None,
    symbols: Tuple[str, ...] = (),
    bar_volume: Optional[np.ndarray] = None,
) -> OrderPlan:
    """Size the orders that move ``current_units`` to ``target_weights``.

    This is the single definition of order sizing for the whole project. The
    backtest engine and the paper account both call it, so the two cannot drift
    apart again without a test failing.

    Sizing uses the **execution reference price**, not the price the signal was
    formed at. That models a notional (fractional-share) order: you ask for
    $8.33 of a name and the broker fills that dollar amount at whatever the
    opening print is. It is not valid for whole-share orders, where you must
    commit to a share count before the open — see the README.
    """
    target_weights = np.asarray(target_weights, dtype=float)
    current_units = np.asarray(current_units, dtype=float)
    reference_price = np.asarray(reference_price, dtype=float)
    n = target_weights.shape[0]
    live = (
        np.ones(n, dtype=bool)
        if tradeable is None
        else np.asarray(tradeable, dtype=bool)
    )
    live = live & np.isfinite(reference_price) & (reference_price > 0)

    safe_price = np.where(live, np.maximum(reference_price, 1e-12), 1.0)
    target_units = np.where(live, target_weights * equity / safe_price, current_units)

    delta = target_units - current_units
    notional = np.abs(delta) * np.where(live, reference_price, 0.0)

    # Dust is measured against the position being adjusted, not the account: in
    # a 20-name book each position is ~5% of equity, so an equity-relative floor
    # would block every trade the strategy wants to make. A full exit is never
    # dust — suppressing one leaves a stale holding open after the signal
    # has gone flat.
    full_exit = (target_units == 0.0) & (current_units != 0.0)
    position_ref = np.maximum(np.abs(target_units), np.abs(current_units)) * np.where(
        live, reference_price, 0.0
    )
    suppressed = (notional < model.min_trade_frac * position_ref) & ~full_exit
    delta = np.where(suppressed | ~live, 0.0, delta)
    notional = np.abs(delta) * np.where(live, reference_price, 0.0)

    participation = np.zeros(n)
    if bar_volume is not None and model.costs.impact_bps_at_full > 0:
        volume_notional = np.asarray(bar_volume, dtype=float) * np.where(live, reference_price, 0.0)
        participation = np.divide(
            notional, volume_notional,
            out=np.zeros(n), where=np.isfinite(volume_notional) & (volume_notional > 0),
        )

    side = np.sign(delta)
    fill = np.where(live, model.costs.fill_price(reference_price, side, participation), np.nan)

    return OrderPlan(
        symbols=tuple(symbols) if symbols else tuple(f"a{i}" for i in range(n)),
        delta_units=delta,
        reference_price=reference_price,
        fill_price=fill,
        notional=notional,
        is_full_exit=full_exit,
        suppressed=suppressed,
    )
