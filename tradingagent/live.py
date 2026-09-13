"""Turn a fitted agent into today's instruction.

A backtest that cannot tell you what to do this morning is a research toy. This
module answers exactly one question - *given everything up to the most recent
closed bar, what position should the account be holding now?* - and shows the
working, so the number can be sanity-checked before any money moves.

Nothing here places orders. It prints a target position; a human executes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .agent import AgentConfig, TradingAgent
from .data import bars_per_year, load_prices
from .risk import RiskConfig


@dataclass
class Recommendation:
    symbol: str
    as_of: pd.Timestamp
    price: float
    raw_signal: float          # conviction in [-1, 1]
    target_weight: float       # after the risk layer, x equity
    equity: float
    target_notional: float
    target_units: float
    current_units: float
    trade_units: float
    components: Dict[str, float]
    blend: Dict[str, float]

    def __str__(self) -> str:
        side = "LONG" if self.target_weight > 0 else ("SHORT" if self.target_weight < 0 else "FLAT")
        action = (
            "hold"
            if abs(self.trade_units * self.price) < 0.02 * self.equity
            else ("BUY" if self.trade_units > 0 else "SELL")
        )
        lines = [
            f"== {self.symbol} - as of {self.as_of} ==",
            f"  last close        ${self.price:,.2f}",
            f"  conviction        {self.raw_signal:+.2f}  (-1 fully short .. +1 fully long)",
            f"  target position   {side} {abs(self.target_weight):.2f}x equity "
            f"= ${self.target_notional:,.2f} ({self.target_units:.6f} units)",
            f"  currently holding {self.current_units:.6f} units",
            f"  action            {action} {abs(self.trade_units):.6f} units",
            "  what each model says:",
        ]
        for name, val in sorted(self.components.items(), key=lambda kv: -abs(kv[1])):
            lines.append(f"    {name:<20} signal {val:+.2f}   weight {self.blend.get(name, 0.0):.2f}")
        return "\n".join(lines)


def recommend(
    df: pd.DataFrame,
    *,
    symbol: str = "asset",
    equity: float = 100.0,
    current_units: float = 0.0,
    agent_config: AgentConfig | None = None,
    risk_config: RiskConfig | None = None,
) -> Recommendation:
    """What to hold right now, given history up to the last closed bar."""
    agent = TradingAgent(agent_config or AgentConfig(), risk_config or RiskConfig())
    sized = agent.sized_weight(df)
    raw = agent.diagnostics_["combined"]["combined"]
    signals = agent.diagnostics_["signals"]
    blend = agent.diagnostics_["blend_weights"]

    price = float(df["close"].iloc[-1])
    weight = float(sized.iloc[-1])
    notional = weight * equity
    target_units = notional / price if price > 0 else 0.0
    return Recommendation(
        symbol=symbol,
        as_of=df.index[-1],
        price=price,
        raw_signal=float(raw.iloc[-1]),
        target_weight=weight,
        equity=float(equity),
        target_notional=notional,
        target_units=target_units,
        current_units=float(current_units),
        trade_units=target_units - float(current_units),
        components={c: float(signals[c].iloc[-1]) for c in signals.columns},
        blend={c: float(blend[c].iloc[-1]) for c in blend.columns},
    )


def recommend_symbol(
    symbol: str = "BTC-USD",
    *,
    interval: str = "1d",
    start: str = "2017-01-01",
    source: str = "coinbase",
    equity: float = 100.0,
    current_units: float = 0.0,
    agent_config: AgentConfig | None = None,
    risk_config: RiskConfig | None = None,
    refresh: bool = True,
) -> Recommendation:
    """Download fresh data for ``symbol`` and recommend a position."""
    df = load_prices(symbol, interval=interval, start=start, source=source, refresh=refresh)
    cfg = agent_config or AgentConfig(periods_per_year=bars_per_year(interval))
    return recommend(
        df,
        symbol=symbol,
        equity=equity,
        current_units=current_units,
        agent_config=cfg,
        risk_config=risk_config,
    )


# --------------------------------------------------------------------------- #
# cross-sectional
# --------------------------------------------------------------------------- #
@dataclass
class BasketRecommendation:
    """Today's target basket from a cross-sectional ranker."""

    as_of: pd.Timestamp
    equity: float
    positions: pd.DataFrame        # symbol, score, rank, weight, dollars, shares
    n_live: int
    long_only: bool
    model: str

    def __str__(self) -> str:
        longs = self.positions[self.positions["weight"] > 0]
        shorts = self.positions[self.positions["weight"] < 0]
        lines = [
            f"== basket as of {self.as_of.date()} - {self.model} ==",
            f"  universe          {self.n_live} names ranked",
            f"  account           ${self.equity:,.2f}",
            f"  book              {len(longs)} long"
            + (f", {len(shorts)} short" if len(shorts) else " (long only)"),
            "",
            f"  {'symbol':<8}{'rank':>6}{'score':>9}{'weight':>9}{'dollars':>11}{'shares':>12}",
        ]
        for _, row in self.positions.iterrows():
            lines.append(
                f"  {row['symbol']:<8}{int(row['rank']):>6}{row['score']:>9.2f}"
                f"{row['weight']:>8.1%}{row['dollars']:>11,.2f}{row['shares']:>12.4f}"
            )
        return "\n".join(lines)


def recommend_basket(
    panel,
    *,
    ranker=None,
    rules=None,
    equity: float = 100.0,
    features: Optional[list] = None,
    periods_per_year: float = 252.0,
) -> BasketRecommendation:
    """Rank the universe on the latest bar and size the basket.

    Uses the most recent complete bar, so this is what to hold from the next
    open. Fractional share counts are given because a $100 account cannot buy
    whole shares of most large caps - a broker offering fractional shares is a
    hard requirement for running this at that size.
    """
    from .cross_section import EqualBlendRanker, PortfolioRules, scores_to_weights
    from .features import feature_panel

    ranker = ranker or EqualBlendRanker()
    rules = rules or PortfolioRules()
    feats = feature_panel(panel, features, periods_per_year=periods_per_year)
    scores = ranker.score(feats)
    weights = scores_to_weights(scores, rules)

    as_of = panel.index[-1]
    row_w = weights.loc[as_of]
    row_s = scores.loc[as_of]
    held = row_w[row_w.abs() > 1e-9]
    price = panel.close.loc[as_of]

    frame = pd.DataFrame(
        {
            "symbol": held.index,
            "score": row_s.reindex(held.index).to_numpy(),
            "weight": held.to_numpy(),
            "dollars": (held * equity).to_numpy(),
            "shares": (held * equity / price.reindex(held.index)).to_numpy(),
        }
    )
    frame["rank"] = row_s.rank(ascending=False).reindex(frame["symbol"]).to_numpy()
    frame = frame.sort_values("weight", ascending=False).reset_index(drop=True)
    frame = frame[["symbol", "rank", "score", "weight", "dollars", "shares"]]

    return BasketRecommendation(
        as_of=as_of,
        equity=float(equity),
        positions=frame,
        n_live=int(row_s.notna().sum()),
        long_only=bool(rules.long_only),
        model=getattr(ranker, "name", type(ranker).__name__),
    )
