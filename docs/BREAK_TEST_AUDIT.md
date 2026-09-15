# Break-test audit — current state before the falsification work

*Phase 1. What this repository already gets right, what is still weak, and which experiments are
worth running in what order.*

This is written to be used, not admired. The ranked table in §10 is the operative part.

---

## 1. Architecture as it actually stands

```
tradingagent/
  data.py          4 sources (Coinbase, Yahoo chart, Nasdaq, CSV) + synthetic simulator, disk cache
  universe.py      Panel (wide OHLCV), named universes, MembershipProvider interface
  features.py      cross-sectional features, purging, causal standardisation
  indicators.py    SMA/EMA/RSI/ATR/ADX/Donchian/Bollinger/percentile-rank
  strategies.py    6 single-asset strategies + BuyAndHold, each with a param_space
  agent.py         ensemble: causal performance-weighted blend of strategies
  risk.py          vol targeting, ATR stops, drawdown kill switch, (disabled) Kelly
  execution.py     THE cost + fill model. CostModel, ExecutionModel, plan_rebalance
  engine.py        bar-by-bar backtest loop; compounding, stops, financing
  paper.py         forward paper account; same execution module
  optimize.py      single-asset walk-forward, purge + embargo, search space
  xs_optimize.py   cross-sectional walk-forward, ridge/blend rankers
  timescale.py     calendar-time rescaling of every bar-count parameter
  sweep.py         interval x frequency grid, frictionless twin
  robustness.py    parameter stability, regime classification, era report, acceptance gate
  holdout.py       reserved-era guard + on-disk use counter
  ledger.py        append-only research ledger
  metrics.py       statistics, deflated Sharpe, block bootstrap
  survivorship.py  delisting simulation
```

Execution timing is canonical and single-sourced: a decision at the close of bar `t` fills at the
**open of bar `t+1`**, priced through `execution.plan_rebalance`, which both `engine.py` and
`paper.py` call. That is the single most important correctness property in the repository and it
is genuinely centralised.

---

## 2. What is already correct — do not redo this

| Area | Status | Evidence |
|---|---|---|
| Look-ahead, indicator → strategy → agent → risk → engine → walk-forward | **Solid** | `tests/test_no_lookahead.py`, 75 tests, **future-poisoning** (replace the future with garbage, assert the past is bit-identical) rather than truncation, with negative controls that fail if poisoning stops biting |
| Backtest ↔ paper parity | **Solid** | `tests/test_execution_parity.py`, exact equity reconciliation under every cost scenario; found 4 real bugs when written |
| Cost model | **Solid** | Adverse-fill pricing, spread/slippage/impact/financing/borrow, three scenarios, square-root impact |
| Walk-forward hygiene | **Solid** | Purge + embargo, 45% training-drawdown disqualifier, top-k blending, equity chained across folds |
| Multiple-testing correction | **Implemented** | Deflated Sharpe, block bootstrap CIs |
| Search-space discipline | **Good** | Reduced 3.3e9 → 2,304; nuisance knobs pinned, economic knobs searched |
| Holdout protection | **Good** | `Holdout.guard` raises on reserved data; every look is counted on disk |
| Interval rescaling | **Solid** | `timescale.py`, with a test that fails if any new integer knob is left unclassified |
| Regime classification | **Correct** | Causal trailing windows, not full-sample quantiles |
| Paper trading restrictions | **Intact** | Long-only, no leverage beyond configured cap, no live order submission anywhere |

---

## 3. Remaining methodological weaknesses

Ordered by how much they threaten the validity of a conclusion.

### 3.1 The research ledger is not wired into the code that searches — **CRITICAL**

`ledger.py` exists, is tested, and records candidate counts. But:

```
  experiments recorded   1
  used for selection     0
  effective trials       0
```

The interval/frequency study evaluated **15,808 configurations** and recorded **none of them**.
`sweep.py` never passes `ledger=` to `walk_forward`, and neither do the four sweep scripts. The
multiple-testing total the repository can *prove* is 0; the true total is at least 15,808 plus
everything the earlier sessions ran.

This is the worst kind of weakness because the tooling is already built and simply is not
connected. Any claim of the form "corrected for N trials" is currently correct only because a
human happened to type N into a report.

### 3.2 No null / placebo / randomization tests at all — **CRITICAL**

There is no experiment anywhere that asks *"what would this machinery produce on data that
contains no edge?"* Every result in the repository is compared against buy-and-hold or against
another strategy. Nothing is compared against noise.

Without this, a deflated Sharpe of 0.13 is a number whose calibration is untested. A shuffled-signal
or block-bootstrapped-returns null gives the sampling distribution of the whole pipeline — search,
selection, walk-forward and all — which is the only way to know whether the pipeline manufactures
apparent edge from nothing.

### 3.3 Point-in-time membership exists but nothing uses it — **HIGH**

`MembershipProvider` / `StaticMembership` / `PointInTimeMembership` / `apply_membership` are
written and tested. But no walk-forward accepts a provider, `load_panel` does not take one, and
the ledger has no field for which membership a result used. The interface is honest; its
disconnection means a survivorship-biased result and a clean one are indistinguishable in the
record.

### 3.4 Cross-asset validation is absent — **HIGH**

Every crypto conclusion rests on BTC-USD. Every stock conclusion rests on one 125-name universe.
Nothing tests whether the mechanism generalises to an asset that was never involved in developing
it. This is the cheapest strong evidence available and it has not been collected.

### 3.5 Era analysis is too thin to falsify anything — **MEDIUM**

`era_report` returns return / Sharpe / worst-bar per year. It does not report drawdown, drawdown
duration, trade count, exposure, benchmark or excess return, so it cannot answer "does this only
work in one regime?" — which is the question that matters most for a strategy whose entire history
contains two crypto bull markets.

### 3.6 Baselines are weak — **MEDIUM**

Buy-and-hold and an equal-weight basket exist. There is no simple-moving-average baseline and no
single-parameter momentum baseline evaluated under identical costs and windows. The repository can
say "the agent beat buy-and-hold"; it cannot say "the agent beat a 200-day SMA rule", which is the
comparison that decides whether the ensemble's complexity is earning anything.

### 3.7 Capacity and participation are modelled but never reported — **MEDIUM**

`CostModel` has `impact_bps_at_full` and a `participation` argument, but no report shows
participation rate, average trade size, or largest trade against volume. On a $100 account this is
genuinely harmless; the framework nonetheless cannot currently answer the question at any size.

### 3.8 Statistical reporting mixes sample types — **MEDIUM**

`full_report` is comprehensive on statistics but does not label a result as IN-SAMPLE /
OUT-OF-SAMPLE / HOLDOUT / UNSEEN-ASSET / UNSEEN-PERIOD. Two numbers with very different epistemic
status print identically.

### 3.9 Known data limits — **NOT FIXABLE HERE, MUST STAY VISIBLE**

| Limit | Consequence |
|---|---|
| Nasdaq is split- but **not dividend-adjusted** | price returns, not total returns; systematically penalises high-yield names |
| Nasdaq caps at ~10 years | no 2007-2009 crisis coverage for stocks |
| Yahoo is HTTP 429 from this IP | no total-return equity data, no intraday equity data |
| Coinbase BTC starts 2015-07 | no 2007-2012 crypto data exists at all |
| Coinbase 5m = 833 days, 1m = 129 days | too short for one walk-forward fold |
| `US_LARGE_CAP` is today's list | survivorship-biased by construction |

**The requested 2007-2009, 2010-2012 and 2013-2015 eras are largely unavailable.** Crypto does not
exist before 2015 and the working equity source starts 2016. This will be stated as missing
coverage, not fabricated.

---

## 4. Potential sources of false alpha, ranked

1. **Two crypto bull markets.** 2017 and 2020-21 dominate the entire BTC record. The daily agent
   is already known to fail on every start date after 2020.
2. **Search luck.** Demonstrated inside this repository: the same 6h cell moved $442 → $886 on
   candidate budget alone, and the hourly cell moved $497 → $2,199 on the random seed alone.
3. **Survivorship in the stock universe.** Long-only results over a today's-winners list are an
   upper bound.
4. **Price-return vs total-return** on Nasdaq data, tilting the cross-section against high-yield
   sectors.
5. **The dust filter as a hidden free parameter.** `min_trade_frac=0.10` was shown to be
   non-optimal at 6h; it has never been searched, so it is an unaccounted degree of freedom.
6. **Selection across the cell grid itself.** Picking the best interval/cadence out of 13 cells is
   a search, and only the sweep-wide deflated Sharpe accounts for it.

---

## 5. Execution assumptions, and how realistic each is

| Assumption | Realistic? | Note |
|---|---|---|
| Fill at next open | Yes, conservative | Real fills can be better or worse; no partial-fill model |
| Fee + half-spread + slippage as adverse price | Yes | Arithmetically identical to a cash fee, and fixes the trade log |
| Square-root impact | Reasonable | Only bites when `participation` is passed, which no report does |
| Financing on leveraged exposure | Yes | Charged per bar |
| Fractional shares | Yes | Required at $100; genuinely available retail |
| No partial fills | Optimistic | Assumes the whole order fills at one price |
| No borrow-availability constraint | Optimistic, but shorting is off by default | |
| No slippage asymmetry in crashes | **Optimistic** | Costs are constant; real spreads widen exactly when the kill switch fires |

The last row is the most material and is not currently testable.

---

## 6. Survivorship-bias risk

* **Crypto (BTC-USD single asset): none.** One asset, still trading.
* **Stocks: material and unquantified.** `US_LARGE_CAP` deliberately includes laggards (INTC, WBA,
  GE, PARA, F …) which blunts the bias, and a cross-sectional ranker is less exposed than a
  long-only buyer, but neither removes it.
* `survivorship.py` can *simulate* delistings; it has never been run against the headline result.
* Fully solving this needs point-in-time index membership with entry/exit dates plus delisted-name
  price history — Norgate, Sharadar, CRSP. None is reachable from here.

---

## 7. Regime limitations

BTC history = 2015-07 → 2026-09. It contains two bull markets, two ~80% drawdowns, and no period
resembling a conventional equity bear market. The stock panel = 2016-09 → 2026-09: one COVID crash,
one 2022 drawdown, no 2008.

**Neither dataset contains a prolonged, slow bear market.** That is the regime trend-following is
most likely to fail in, and it is absent from both.

---

## 8. Statistical weaknesses

* Deflated Sharpe is implemented but its **calibration is untested** (see §3.2).
* Bootstrap CIs exist but are not attached to every headline figure.
* Seed counts are small — 4 seeds where hourly's spread is 4.4×.
* No correction for the fact that the *same data* has now been examined across many sessions.

---

## 9. Optimizer / selection risks

* Ledger disconnected (§3.1) — the dominant risk.
* `Holdout.guard` must be passed explicitly; nothing forces it, and the sweep never did.
* The objective `target_growth` contains a hand-set drawdown penalty (`8.0 * excess_dd²`) and a
  thin-trade penalty — both unsearched constants that shape every selection.
* `top_k=5` blending is a fixed choice that was never justified against alternatives.

---

## 10. Ranked experiment list

Scored 1-5. **Priority** = research value × expected impact ÷ (bias risk × cost), judgementally.

| # | Experiment | Research value | Expected impact | Bias risk | Compute | Priority |
|---|---|---|---|---|---|---|
| 1 | **Null / placebo suite** — shuffled signals, block-bootstrapped returns, random-parameter control, run through the *entire* pipeline | 5 | 5 | 1 | 3 | **Do first** |
| 2 | **Wire the ledger into every search path** + record membership provenance | 5 | 4 | 1 | 1 | **Do first** |
| 3 | **Cross-asset validation on frozen parameters** (SPY/QQQ/IWM/DIA + ETH, unseen) | 5 | 4 | 2 | 3 | **Do first** |
| 4 | **Simple baselines under identical costs** (SMA, single-parameter momentum, cash) | 4 | 4 | 1 | 2 | High |
| 5 | **Era + regime decomposition with full statistics** | 4 | 3 | 1 | 2 | High |
| 6 | **Parameter perturbation / stability report** on the frozen set | 4 | 3 | 1 | 2 | High |
| 7 | **Cost-adversity ladder to the break-even point** (esp. hourly) | 3 | 3 | 1 | 2 | Medium |
| 8 | **PIT universe interface wired + leakage test** | 4 | 2 | 1 | 1 | Medium |
| 9 | **Capacity / participation reporting** | 2 | 1 | 1 | 1 | Low |
| 10 | **Delisting simulation against the headline** | 3 | 2 | 2 | 2 | Medium |

Explicitly **not** doing: open-ended parameter optimisation, searching for a $1,000 configuration,
tuning `min_trade_frac`, or adding assets because they improve the curve.

---

## 11. The single question this audit exists to enable

> *What experiment would make us conclude this strategy probably does NOT have a durable edge?*

Concretely, any one of these:

1. The null suite produces deflated Sharpes indistinguishable from the real strategy's.
2. Frozen parameters fail on every unseen asset.
3. Performance concentrates in one regime and vanishes elsewhere.
4. A 200-day SMA baseline matches the ensemble under identical costs.
5. Parameter perturbation collapses the result.

Experiments 1-5 of §10 are designed to make each of those outcomes *possible*. If none of them
fires, the evidence is meaningfully stronger. If any fires, the correct conclusion is that the
edge is not established — and that is a successful outcome for this exercise.
