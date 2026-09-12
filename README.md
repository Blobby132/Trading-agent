# Trading agent: $100 -> $1,000, backtested honestly

A compounding trading agent for a small account, with a backtest framework built around the
one question that matters for a $100 stake: **can it reach $1,000, how long does it take, and
what does it have to survive on the way?**

The hard part of this problem is not writing a strategy. It is writing a backtest that does not
lie to you. Two things make a backtest lie, and both are addressed structurally here rather than
by good intentions:

| The lie | What it looks like | What this repo does about it |
|---|---|---|
| **Lookahead** | using a price the decision could not have seen | a signal formed at bar `t` is filled at the **open of bar `t+1`**; every indicator, strategy and blend is proven causal by a test that truncates the future and checks the past does not move |
| **Overfitting** | trying 500 settings, reporting the winner | parameters are chosen on a **training window** and traded untouched on the **next** window; only the traded windows are reported |

Costs are charged like an exchange charges them - fee and slippage on every fill, financing on
leverage, borrow on shorts. On a $100 account that is not a detail; it is the main thing standing
between the account and the target.

> **Not financial advice.** This is research code. It backtests and it prints a recommended
> position; it never places an order. Simulated results do not predict future returns. Crypto
> routinely falls 80%. Do not trade money you cannot afford to lose completely.

---

## Quick start

### Google Colab (the intended way)

Open `notebooks/Trading_Agent_Backtest.ipynb` in Colab and run all cells. The first cell clones
this repo and installs dependencies; everything else runs top to bottom with no API keys and no
account - market data comes from Coinbase's public candle endpoint.

```
https://colab.research.google.com/github/Blobby132/Trading-agent/blob/claude/trading-agent-backtest-sih2te/notebooks/Trading_Agent_Backtest.ipynb
```

### Locally

```bash
pip install -r requirements.txt

# walk-forward backtest of a $100 account chasing $1,000
python -m tradingagent.cli --symbols BTC-USD --capital 100 --target 1000

# a portfolio sharing the same $100
python -m tradingagent.cli --symbols BTC-USD ETH-USD LINK-USD --candidates 200

# no network? the synthetic simulator exercises the whole pipeline offline
python -m tradingagent.cli --source synthetic --symbols SYN --mode single

pytest -q          # the causality and accounting tests
```

The CLI writes `results/tearsheet.png`, `results/equity.csv`, `results/folds.csv` and
`results/stats.json`.

---

## What the agent actually is

An **ensemble with adaptive weights**, not a single rule. Six strategies run side by side; at
every bar the agent scores each one on its own trailing, hypothetical P&L - using only
information available at that bar - and blends their signals with a softmax of those scores. A
model that stops working loses funding within weeks without anyone intervening.

```
         data ──► indicators ──► six strategies ──► adaptive blend ──► regime filter
                                                                            │
                                                                            ▼
                                                                       risk layer
                                                      (vol targeting, leverage cap, ATR stops,
                                                       drawdown kill switch)
                                                                            │
                                                                            ▼
                                                                    execution engine
                                                       (fills at next open, fees, slippage,
                                                        financing, compounding, ruin)
```

**The strategies** (`tradingagent/strategies.py`)

| Strategy | Thesis | Shines when |
|---|---|---|
| `ema_trend` | EMA spread, gated by ADX | sustained directional moves |
| `donchian` | channel breakout, shorter exit channel | expansions out of consolidation |
| `ts_momentum` | trailing return, ranked against its own history | persistent drift |
| `bollinger_breakout` | expansion out of a volatility squeeze | regime changes |
| `mean_reversion` | fade stretched z-scores when ADX is low | ranging, choppy markets |
| `rsi_pullback` | buy dips inside an uptrend | shallow corrections in a bull run |

Two of those are counter-trend and four are with-trend on purpose: the blend should have
something to say whichever regime shows up, and the weighting decides which.

**The risk layer** (`tradingagent/risk.py`) turns a direction into a size: volatility targeting
so a quiet regime is not under-traded and a violent one does not wipe the account, a hard
leverage cap, ATR stops, and a drawdown kill switch that flattens the book and stands aside for
a cooldown.

**The engine** (`tradingagent/engine.py`) is a bar-by-bar loop rather than a vectorised dot
product, because three things it must model are path dependent: compounding, intrabar stops, and
that kill switch.

---

## Why the walk-forward is the only number worth quoting

```
|<--- train 730 bars --->|<-embargo->|<-- trade 182 bars -->|
                                   |<--- train 730 bars --->|<-embargo->|<-- trade 182 -->|
                                                          (equity carries across windows)
```

On each training window the optimiser samples configurations - which strategies, how to blend
them, how much volatility to target, how much leverage, how wide the stops - and scores them on
a growth objective that **penalises deep drawdowns quadratically and disqualifies any candidate
that drew down more than 45% in training**, however good its return. The best few are then
*averaged* and traded, untouched, over the following six months. Averaging the top-k rather than
picking the single winner is what stops one lucky parameter cell from becoming the strategy.

The stitched equity curve is a single continuously-compounding account, which is exactly what
the $100 -> $1,000 question is about.

Three further checks ship with it, because a walk-forward can still flatter a search this wide:

- **Seed sensitivity** - the search is random; if the result needs one particular seed, it is not a result.
- **Block bootstrap** (`metrics.monte_carlo_paths`) - resamples the realised out-of-sample returns
  in blocks, preserving streaks and volatility clustering, to estimate `p(reach target)` and,
  more importantly, `p(ruin)`.
- **Deflated Sharpe** (`metrics.deflated_sharpe`) - how much of the Sharpe survives after
  accounting for how many configurations were tried.

---

## Results

<!-- RESULTS -->

---

## Layout

```
tradingagent/
  data.py         Coinbase / Yahoo / CSV loaders, disk cache, synthetic simulator
  indicators.py   causal technical features
  strategies.py   six signal generators + a registry
  agent.py        the ensemble, adaptive weighting, regime filter, portfolio version
  risk.py         volatility targeting, Kelly, ATR stops, drawdown kill switch
  engine.py       execution, costs, leverage, compounding, ruin
  optimize.py     walk-forward search, objectives, top-k blending
  metrics.py      statistics, time-to-target, bootstrap, deflated Sharpe
  report.py       tearsheet plots
  live.py         "what should I hold right now"
  cli.py          command-line entry point
notebooks/        the Colab notebook
tests/            84 tests, mostly about causality and accounting
```

### Configuration

Everything is a dataclass, so nothing is hidden in a function signature:

```python
from tradingagent.agent import AgentConfig, TradingAgent
from tradingagent.engine import ExecutionConfig
from tradingagent.risk import RiskConfig

agent = TradingAgent(
    AgentConfig(
        strategies=["ema_trend", "donchian", "ts_momentum"],
        weighting="adaptive",     # "equal" | "best"
        perf_lookback=120,        # bars of trailing P&L used to score a strategy
        regime_filter=True,       # veto signals that fight the long-term trend
        signal_smooth=5,
    ),
    RiskConfig(
        target_vol=0.50,          # annualised
        max_leverage=2.0,
        atr_stop_mult=6.0,        # wide on purpose - see below
        max_drawdown_stop=0.35,   # flatten and stand aside
    ),
)
result = agent.backtest(prices, ExecutionConfig(initial_capital=100.0, target_equity=1000.0))
print(result.stats())
```

Three defaults that were set by measurement rather than convention:

- **`atr_stop_mult=6.0`.** A 2-3 ATR stop sits inside daily crypto noise; it gets shaken out of
  precisely the trends the trend models exist to capture. Tight stops measurably *cost* money here.
- **`min_trade_frac=0.10`.** On a $100 account, rebalancing for the sake of a 2% weight change is
  pure fee. Ignoring small adjustments cut turnover by more than half.
- **`signal_smooth=5`.** Damps day-to-day flip-flopping in the blend, which is otherwise paid for
  in slippage.

### The kill switch bug worth knowing about

An early version latched: once equity fell past the drawdown limit it went flat, but the
high-water mark never reset, so every subsequent bar re-armed the cooldown and the agent never
traded again. The fix - restart the drawdown clock from wherever the account actually is when the
cooldown expires - is in `engine.py`, and `tests/test_engine.py::test_drawdown_kill_switch_flattens_then_resumes`
exists to keep it fixed.

---

## Honest limitations

- **The asset did the heavy lifting.** BTC rose roughly 100x over this sample. Any long-biased
  agent looks good against that backdrop. Always read the buy-and-hold column next to the agent's -
  buy & hold frequently wins on return while losing badly on drawdown, and that difference is the
  actual claim being made here.
- **I iterated on BTC.** The pipeline was developed while looking at BTC daily bars, so BTC results
  carry my own selection bias no walk-forward can remove. The cross-asset section of the notebook
  is the corrective: the same code, unchanged, pointed at markets it was never tuned on.
- **Execution is optimistic.** Fills at the open with fixed slippage; real fills on thin pairs are
  worse, and a gap through a stop is worse again.
- **A 10x needs volatility, and volatility is symmetric.** Every configuration that reaches $1,000
  quickly is also a configuration that can reach $0. That is why the bootstrap reports `p(ruin)`
  next to `p(reach target)`, and why the risk settings should be chosen by reading the first column.
- **No regime is permanent.** The adaptive blend shortens how long a broken model stays funded. It
  cannot manufacture an edge in a regime where nothing in the panel works.
