# Candle interval and trading frequency

*Can the agent reach $1,000 faster by looking at finer candles and trading more often?*

This document is the answer, the method that produced it, and the reasons to distrust it.

The short version is in [Verdict](#verdict). Everything above it is the evidence.

---

## 1. What was swept, and what was held fixed

Two new dimensions:

1. **Candle interval** — 1d, 6h, 1h for the full history; 15m over the shorter window Coinbase
   serves it for; 5m and 1m audited but **not backtested**, for a reason given in §3.
2. **Trading frequency** — how often the book is allowed to move, swept independently of how
   often the strategy forms a view.

Everything else is the machinery the daily backtest already used, unchanged:

| Safeguard | Status |
|---|---|
| Signal at bar `t` fills at the **open of bar `t+1`** | unchanged, and re-audited at every interval |
| Walk-forward: fit on one window, trade untouched on the next | unchanged |
| Purge + embargo between fit and trade | unchanged, rescaled to the same calendar span |
| 45% max-training-drawdown disqualifier | unchanged |
| Deflated Sharpe / bootstrap / robustness battery | unchanged, and now deflated across the **whole** sweep |
| Transaction costs (fee, spread, slippage, impact, financing, borrow) | unchanged — BASE, 15 bps/side |

No parameter was relaxed to make a faster configuration look better. Where a result is weak,
§7 says so.

---

## 2. Rescaling: the thing that makes this comparison mean anything

Every lookback in this repository is a **bar count**, and every default was chosen against
**daily** bars. That assumption is invisible until the interval changes, and then it quietly
replaces the strategy with a different one.

`mom_lookback=60` means *sixty days* on daily candles and *sixty hours* — two and a half days —
on hourly candles. Running the daily-tuned configuration on hourly data without rescaling is not
"the same strategy, sampled faster". It is a much faster strategy, and any difference in its
result tells you nothing about the interval.

So `tradingagent/timescale.py` converts every bar count to the same **calendar span**. Here is
the full audit — the daily value, and what each interval actually ran with:

### Search-space axes

| parameter | daily | 6h | 1h | 15m |
|---|---|---|---|---|
| `perf_lookback` | 60, 250 | 240, 1000 | 1440, 6000 | 5760, 24000 |
| `regime_trend` | 200 | 800 | 4800 | 19200 |
| `signal_smooth` | 1, 5 | 4, 20 | 24, 120 | 96, 480 |
| `ema_fast` | 20 | 80 | 480 | 1920 |
| `ema_slow` | 100 | 400 | 2400 | 9600 |
| `donchian_entry` | 40 | 160 | 960 | 3840 |
| `mom_lookback` | 60 | 240 | 1440 | 5760 |

### Walk-forward windows — the ones that matter most

| parameter | daily | 6h | 1h | 15m |
|---|---|---|---|---|
| `train_bars` | 730 | 2,920 | 17,520 | 70,080 |
| `test_bars` | 182 | 728 | 4,368 | 17,472 |
| `embargo_bars` | 10 | 40 | 240 | 960 |

730 hourly bars is a *month*. Leaving `train_bars` at 730 would have trained on a month and
still called it a walk-forward. Every interval trains on two years and trades six months.

### Risk and agent windows

| parameter | daily | 6h | 1h | 15m |
|---|---|---|---|---|
| `AgentConfig.perf_lookback` | 120 | 480 | 2,880 | 11,520 |
| `AgentConfig.regime_trend` | 200 | 800 | 4,800 | 19,200 |
| `AgentConfig.signal_smooth` | 5 | 20 | 120 | 480 |
| `RiskConfig.vol_lookback` | 30 | 120 | 720 | 2,880 |
| `RiskConfig.atr_n` | 14 | 56 | 336 | 1,344 |
| `RiskConfig.reentry_lockout_bars` | 3 | 12 | 72 | 288 |
| `RiskConfig.cooldown_bars` | 10 | 40 | 240 | 960 |

Multipliers, thresholds, z-scores and annualised rates are **not** rescaled — `atr_stop_mult`,
`target_vol`, `max_leverage`, `max_drawdown_stop`, `min_trade_frac` are dimensionless or already
per-year, and scaling one would be a bug. The split is written down in `BAR_COUNT_PARAMS` and
`SCALE_INVARIANT_PARAMS` rather than inferred from parameter names, and a test walks the config
dataclasses and fails if any integer knob is missing from both lists.

`periods_per_year` is reset per interval as well (365 → 1,460 → 8,760 → 35,040). Forgetting that
is how an hourly backtest reports a volatility √24 too low and a Sharpe √24 too high.

### What rescaling does **not** buy

It equalises the *horizon*, not the *strategy*. A 480-bar hourly EMA and a 20-bar daily EMA both
average about twenty days of price, but the hourly one is built from intraday prices the daily
one never sees, so it reacts to microstructure the daily series smooths away by construction.
Rescaling removes the trivial explanation for a difference between intervals. It does not remove
every explanation.

---

## 3. Data limits, stated before the results

### Coverage actually downloaded (BTC-USD, Coinbase)

| interval | history available | bars | coverage | largest gap |
|---|---|---|---|---|
| 1d | 2015-07-20 → 2026-09-12 | 4,073 | 100.0% | 1 bar |
| 6h | 2015-07-20 → 2026-09-14 | 16,293 | 100.0% | 2 bars |
| 1h | 2015-07-20 → 2026-09-14 | 97,706 | 99.94% | 16 bars |
| 15m | 2021-01-01 → 2026-09-14 | 199,815 | 99.96% | 26 bars |
| 5m | 2024-06-02 → 2026-09-14 | 239,838 | 99.93% | 78 bars |
| 1m | 2026-05-08 → 2026-09-14 | 185,584 | 99.98% | 12 bars |

Note the third column against the second. **The fine intervals have many bars and very little
history.** 1-minute data is 185,584 bars covering 129 days. That is a large sample of a short
period, which is not the same thing as a large sample.

### Which intervals can even run a walk-forward

One fold needs 730 days of training + 10 days of embargo + 182 days of trading = **922 calendar
days**, whatever the interval, because the windows are rescaled to a fixed calendar span.

| interval | calendar days available | days needed for 1 fold | folds possible | runnable? |
|---|---|---|---|---|
| 1d | 4,072 | 922 | 18 | yes |
| 6h | 4,073 | 922 | 18 | yes |
| 1h | 4,073 | 922 | 18 | yes |
| 15m | 2,082 | 922 | 7 | yes, shorter era |
| 5m | 833 | 922 | **0** | **no** |
| 1m | 129 | 922 | **0** | **no** |

**5-minute and 1-minute bars cannot be walk-forward tested at all from this data source.** Not
"were not" — cannot. There is not enough calendar history for a single train/embargo/test fold.

The only ways to produce a 5m or 1m result would be to shorten the training window (no longer the
same methodology, and no longer comparable to the other rows) or to skip walk-forward entirely
(an in-sample number, which is the thing this repository exists to avoid). Neither is worth doing,
so neither was done. **No 5m or 1m backtest is reported below, at any level of caveat.**

### The arithmetic that settles the fine end anyway

Before any backtest: how big is a typical bar's move, against the cost of one round trip?
BASE costs are 15 bps per side, so a round trip is **30 bps**.

| interval | median abs. move | mean abs. move | median move ÷ round-trip cost | % of bars moving more than one round trip |
|---|---|---|---|---|
| 1d | 143.8 bps | 230.1 bps | **4.79×** | 86.3% |
| 6h | 60.3 bps | 106.5 bps | **2.01×** | 70.0% |
| 1h | 23.7 bps | 42.6 bps | **0.79×** | 42.0% |
| 15m | 12.2 bps | 19.7 bps | **0.41×** | 18.7% |
| 5m | 6.0 bps | 9.2 bps | **0.20×** | 4.4% |
| 1m | 2.2 bps | 3.4 bps | **0.07×** | **0.2%** |

Read the last column. At 1-minute bars, **99.8% of bars do not move far enough to pay for the
round trip that would capture them.** A strategy trading that interval has to be right almost
exclusively on the remaining 0.2%, and capture most of each move, before it earns anything.

At hourly bars the median move is already smaller than the round trip. That is the interval where
the economics turn, and it turns *against* speed.

**What this does not say.** It is not a claim that nobody can trade at these frequencies. Firms
do — as market makers *earning* the spread with maker rebates and colocation, not as takers
paying it. This repository's cost model is a taker paying a 15 bps half-spread-and-fee. The
honest statement is narrow: **a taker-priced trend strategy cannot pay for itself at these
intervals**, not that the intervals are untradeable in principle.

### Breakeven, as a ceiling

If every rebalance replaced the whole book, annual cost as a fraction of equity would be:

| rebalances/day | round trips/year | annual cost if full turnover |
|---|---|---|
| 0.2 | 73 | 22% |
| 1 | 365 | **110%** |
| 2 | 730 | 219% |
| 4 | 1,460 | **438%** |
| 12 | 4,380 | 1,314% |
| 24 | 8,760 | 2,628% |

Real rebalances move less than the whole book — the dust filter suppresses the small ones — so
treat this as the ceiling, not the estimate. Even as a ceiling it frames the question: at four
rebalances a day the strategy must gross 438% a year before it has earned a cent.

---

## 5. The cross-sectional stock agent

### Interval: there is nothing to sweep

The stock side of this repository cannot be run intraday, for a plain reason:

* **Nasdaq** — the source currently working — serves **daily bars only**. `load_nasdaq` raises
  `ValueError("the Nasdaq endpoint only serves daily bars")` on any other interval.
* **Yahoo** publishes intraday equity bars with short retention caps (roughly 730 days at 1h,
  60 days below that, 7 days at 1m). I could not verify those caps from this environment —
  Yahoo is returning HTTP 429 to this IP for every intraday request — so they are quoted as the
  vendor's documented limits, not as something measured here.

Either way there is no multi-year intraday equity history available to this repository, so
**no intraday stock backtest is reported.** The frequency sweep below is on daily bars, which is
the only honest thing available.

### Frequency: rebalance cadence on daily bars

`rebalance_every` is normally one axis of the cross-sectional search space, which means the
optimiser can quietly opt out of a cadence it dislikes. For this sweep it was **pinned per cell**
so that frequency is the experiment rather than a free parameter. Everything else — the ranker,
the horizon, `top_frac`, the caps, the vol target — was searched as usual, long-only, with the
same candidate budget and seed in every cell.

### The pattern-day-trader question

The README says a $100 account is blocked by both the Reg T margin minimum and the
pattern-day-trader rule. That compresses two different rules that apply to two different account
types, and the distinction matters here:

* **PDT (FINRA Rule 4210)** applies to **margin accounts**. Four or more day trades in five
  business days designates the account a pattern day trader and imposes a **$25,000** minimum
  equity requirement. A $100 account so designated is frozen.
* **A cash account is not subject to PDT at all.** It is subject to settlement instead: proceeds
  settle T+1, and buying with unsettled proceeds and then selling before settlement is a Good
  Faith Violation. Three in twelve months restricts the account to settled cash for 90 days.
* **Shorting requires margin** (Reg T, $2,000 floor), which a $100 account cannot meet — so a
  $100 account is necessarily a cash account, and therefore necessarily long-only.

So the binding constraint on a $100 stock account is **T+1 settlement, not PDT**. The practical
ceiling is roughly one full turnover per business day using settled cash — not zero, but not
intraday either.

**Does any configuration tested here breach it? No — and it cannot.** On daily bars the canonical
execution model decides at the close of bar `t` and fills at the open of bar `t+1`. The earliest
possible exit is the open of bar `t+2`. Every position is therefore held overnight by
construction, and **a day trade — buying and selling the same security on the same day — is
structurally impossible at any `rebalance_every ≥ 1` on daily data.** The PDT rule cannot bind on
anything this repository can currently produce for stocks.

The flag the brief asked for would apply to an intraday stock configuration. No such
configuration exists here, because the data to build one does not.

One number below does need reading carefully: the *fills per day* column counts **individual name
fills across a 125-name universe**, not round trips. A daily rebalance touching 12 names is 12
fills and zero day trades.

---
