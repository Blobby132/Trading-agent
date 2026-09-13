# Framework review: findings, changes and what the evidence now supports

Companion to `docs/AUDIT.md`, which contains the pre-change audit. This is what
was done about it and what changed as a result.

---

## A. Executive summary

The framework had **two execution implementations that disagreed about when an
order fills.** The backtest filled at the next bar's open, as its timing
convention requires. Paper trading sized against the signal bar's close and
filled at that same close — so it transacted on information at the instant it
was generated, and every tracking-error number it produced was partly the
framework arguing with itself.

That is fixed, along with eleven other defects. The framework is now materially
harder to fool:

- one canonical execution and cost model, shared by both paths, with **23
  deterministic parity tests** asserting their equity curves match exactly;
- the search space cut from **3.3 billion combinations to 2,304**, which
  *improved* out-of-sample results and roughly halved the seed-to-seed spread;
- the holdout is structurally protected — optimisers raise rather than train on
  reserved data;
- a research ledger makes repeated experimentation visible, because the
  multiple-testing correction that matters uses the searches you ran last week
  too;
- parameter stability, regime decomposition and a robustness battery are now
  first-class tools rather than things to remember to do.

**On the evidence: it is better than it was, and it is still not strong.** The
deflated Sharpe for the single-asset agent is 0.19–0.56 depending on how trials
are counted — it now touches the 0.5 line where before it did not reach it, but
an interval that straddles the threshold is not a demonstration of edge. The
strongest result in the repository remains the simplest strategy in it, on the
reserved era, and the sample it rests on contains almost no bear market.

---

## B. Issues found

Severity as assigned in the audit. "Found by" notes where a defect surfaced.

### CRITICAL

| # | Issue | Found by |
|---|---|---|
| 1 | **Paper trading filled at the signal bar's close**, not the next execution opportunity. Docstring claimed otherwise. | audit |
| 2 | **Two execution implementations** duplicating sizing, dust filtering and cost logic — already drifted. | audit |
| 3 | **Search space of 3.3×10⁹ combinations** sampled 150×/fold over ~3,300 bars. | audit |
| 4 | **No backtest/paper parity test** existed at all. | audit |

### HIGH

| # | Issue | Found by |
|---|---|---|
| 5 | Six different cost assumptions; the benchmark paid 2 bps while strategies paid 15. | audit |
| 6 | `holdout.py` was not wired into either optimiser — nothing stopped training on reserved data. | audit |
| 7 | No research ledger; repeated searching left no trace. | audit |
| 8 | Kelly estimator: `mu/var` from as few as 30 observations, capped at 4×, no shrinkage. | audit |
| 9 | `Panel.min_history` selects universe members using the whole sample. | audit |
| 10 | No confidence intervals, no regime decomposition, no acceptance criteria. | audit |
| 11 | **Paper sized against the fill bar's own close**, which prints after the open it fills at — a look-ahead inside code written to remove look-ahead. | parity tests |
| 12 | **Paper never accrued financing**, making it quietly cheaper than its own backtest. | parity tests |

### MEDIUM

| # | Issue | Found by |
|---|---|---|
| 13 | Market impact charged by the engine but not by paper; the engine's *recorded* cost excluded impact its cash flow included. | parity tests |
| 14 | Partial sells left `avg_price` stale, so per-position P&L drifted. | audit |
| 15 | `total_costs` silently included financing when the trade log was off — the field meant different things depending on a performance flag. | audit |
| 16 | `divergence_report` took its calendar from a default argument that could disagree with the account it judged. | parity tests |
| 17 | Trade log recorded the reference price, understating what fills cost. | audit |
| 18 | Parameter-stability metric scored "plateau" when *every* neighbour beat the selection, hiding an arbitrary choice. | stability tests |
| 19 | ATR recomputed on every engine call even with stops disabled. | audit |

### LOW

| # | Issue |
|---|---|
| 20 | No guard against `min_trade_frac ≥ 1.0`, which silently disables all rebalancing. |
| 21 | Robustness battery crashed on a naive date slicing a tz-aware index. |
| 22 | No CI configuration. |

---

## C. Changes implemented

**`execution.py` (new)** — the canonical model. `CostModel` (fee, half-spread,
slippage, √-law market impact, financing, borrow) with LOW/BASE/HIGH scenarios;
`ExecutionModel` (fill timing, one-bar decision lag); `plan_rebalance` — the
single definition of order sizing, dust filtering and fill pricing, called by
both the engine and the paper account.

**`engine.py`** — refactored onto `plan_rebalance`; costs charged as an adverse
fill price so the trade log records what was paid; financing delegated to the
cost model; ATR computed lazily. **Verified numerically identical** to its
pre-refactor behaviour on a three-way fingerprint.

**`paper.py`** — rebuilt around pending-order state. `plan_orders` decides,
`schedule_orders` queues, `fill_pending` executes against the *next* bar and
raises if asked to fill on the decision bar. Sizes against the last close before
the fill; accrues financing per elapsed bar; charges impact; average price now
correct on partial sells and flips.

**`holdout.py`** — `guard()` raises `HoldoutViolation` naming how many reserved
bars were handed over and what to call instead; `protect()` is the non-raising
variant. Both optimisers accept `holdout=`.

**`ledger.py` (new)** — append-only record of every run, with
`selection_count()` summing candidates across all selecting runs.

**`robustness.py` (new)** — `parameter_stability`, `classify_regimes`,
`regime_report`, `era_report`, `robustness_battery`, `check_acceptance`.

**`risk.py`** — Kelly gains a sample-size floor, t-statistic shrinkage and a
cap of full Kelly. Still disabled by default.

**`universe.py`** — `MembershipProvider` / `StaticMembership` /
`PointInTimeMembership` / `apply_membership`. Point-in-time membership needs no
engine change: a non-member becomes NaN, which the engine already treats as
untradeable.

**`metrics.py`** — drawdown duration and time underwater, downside volatility,
round-trip reconstruction for real trade-level expectancy, gross vs net return,
cost components, block-bootstrap confidence intervals, and a long-form report
that says plainly when the Sharpe interval includes zero.

**`optimize.py`** — default space reduced to nine economic parameters;
`LEGACY_WIDE_SEARCH_SPACE` retained.

**`cli.py`** — `--mode robustness`, `--cost-scenario`, `--fill-at`.

---

## D. Tests added

| File | Count | Covers |
|---|---|---|
| `test_execution_parity.py` | 23 | exact fills on hand-checkable bars, one-bar lag, backtest/paper parity under every cost scenario, gaps, cost arithmetic, impact, dust, full exits, stops, accounting, calendar mismatch |
| `test_ledger_and_holdout_guard.py` | 15 | optimiser refuses reserved data, ledger records and accumulates trials, corrupt-line tolerance, numpy serialisation |
| `test_robustness.py` | 22 | neighbourhood inference, plateau vs spike detection, causal regime labels, battery pass/fail/error handling |
| `test_kelly.py` | 9 | disabled by default, sample floor, noise produces no leverage, real edge still sized, causality |
| additions to `test_universe.py`, `test_metrics.py`, `test_cli.py` | 9 | point-in-time membership, cost-field semantics, robustness CLI, cost scenario reaching the engine |

**Total: 329 passing, 2 skipped** (was 253).

---

## E. Test results

```
$ pytest -q
329 passed, 2 skipped
```

Engine invariance across the refactor, on a three-scenario fingerprint
(single-asset with stops and kill switch; multi-asset with a delisting;
levered with shorts and financing):

```
single_equity_hash   42273.051994153   IDENTICAL
multi_equity_hash    66108.822241832   IDENTICAL
levered_equity_hash  17174.067347820   IDENTICAL
```

---

## F. Before / after performance

**BTC single-asset walk-forward, out of sample 2017-07 → 2026-09, five seeds.**
The only deliberate change to assumptions is the search space.

| | Legacy space (3.3×10⁹) | Economic space (2,304) |
|---|---|---|
| Median final equity | $1,026 | **$1,141** |
| Seed spread | $706 – $1,390 | **$1,009 – $1,289** |
| Median Sharpe | 0.84 | **0.90** |
| Median max drawdown | −52.3% | **−48.8%** |
| Deflated Sharpe | 0.15 – 0.44 | **0.19 – 0.56** |

Searching less produced **better** out-of-sample results and **halved the
seed-to-seed spread**. That is the expected direction — a smaller space has
fewer ways to fit this particular sample — but it is worth stating that the
change was made for methodological reasons and the performance improvement was
a consequence, not the goal.

Costs, benchmark rebasing and the execution refactor moved nothing, by
construction and by test.

---

## G. Before / after robustness

Cross-sectional momentum (top decile, monthly), 124 US large caps, out of
sample 2019-10 → 2026-09.

**Parameter stability** — new capability; nothing comparable existed before.

| Parameter | Selected | Nearby | Median | Worst | Stability | Verdict |
|---|---|---|---|---|---|---|
| `top_frac` | 0.10 | 0.075–0.125 | 1.04 | 1.00 | 0.92 | plateau |
| `rebalance_every` | 21 | 16–26 | 1.05 | 1.01 | 0.93 | plateau |
| `max_weight` | 0.15 | 0.11–0.19 | 1.09 | 1.09 | 1.00 | plateau (inert) |
| `min_positions` | 5 | 4–6 | 1.09 | 1.09 | 1.00 | plateau (inert) |
| `factor` | mom_12_1 | 4 momentum definitions | 0.83 | 0.72 | 0.66 | acceptable |

No parameter sits on a spike. Two are *inert* — at 124 names the cap and the
position floor never bind — which is honest to record: a stability score of 1.00
on a parameter that does nothing is not evidence of robustness.

**Robustness battery** — 11 of 12 scenarios passed; the twelfth was a harness
timezone bug, since fixed.

| Scenario | Final | Sharpe | Max DD |
|---|---|---|---|
| base | $656 | 1.09 | −32.7% |
| costs: low | $695 | 1.11 | −32.6% |
| costs: high | $594 | 1.04 | −32.8% |
| slippage +25 bps | $579 | 1.02 | −32.8% |
| spread +15 bps | $609 | 1.05 | −32.7% |
| start 2022 | $319 | 1.06 | −22.6% |
| fill at next close | $714 | 1.13 | −30.4% |

Costs barely matter — triple them and Sharpe falls 1.09 → 1.04 — because
turnover is low. The strategy does not depend on the open-fill assumption
either; filling at the next close is slightly *better*.

**Regimes** — where it works and where it does not:

| Regime | Share of sample | Total return | Sharpe | Excess vs benchmark |
|---|---|---|---|---|
| bull | 47.6% | +226% | 1.48 | +129% |
| sideways | 38.9% | +32% | 0.53 | +11% |
| bear | **2.1%** | +5% | 1.12 | +9% |
| high vol | 21.5% | +29% | 0.63 | +23% |
| **benchmark drawdown > 20%** | **1.6%** | **−19%** | **−1.54** | — |

It earns in trends, is mediocre sideways, and **loses in crashes** — which is
what a long-only momentum strategy is supposed to do. Positive in all eight
calendar years, weakest 2022 (+3.3%).

**The caveat that matters more than the table:** 2.1% of bars in a bear trend
and 1.6% in a deep drawdown. Any claim about bear-market behaviour rests on
roughly 36 and 27 bars respectively. That is not evidence, it is an anecdote.

---

## H. Remaining limitations

1. **The universe is still not point-in-time.** The interface exists; the data
   does not.
2. **The sample has almost no bear market.** Structural, and no amount of
   testing fixes it.
3. **Market impact is unvalidated.** The √-law is a reasonable default shape,
   not a calibrated estimate.
4. **Deflated Sharpe still straddles 0.5.** Better than before, not conclusive.
5. **Dividends are missing from the Nasdaq fallback**, tilting the
   cross-section against high-yield names.
6. **No CI.** The suite runs manually.
7. **The paper account holds no stops or kill switch.** It matches the *current*
   long-only strategy exactly; a stop-using strategy would diverge, and nothing
   yet detects that.

---

## I. Could not be fixed — data or infrastructure

- **Point-in-time index membership** — requires a vendor (Norgate, Sharadar,
  CRSP). Interface built, documented, unused.
- **Delisted-name history on the Nasdaq source** — structurally impossible; it
  is a live-quote API. Yahoo serves these, and was rate-limiting the machine
  these measurements were taken on.
- **Total-return data on the fallback source** — Nasdaq publishes
  split-adjusted prices only.
- **Intraday fills, queue position, partial fills** — needs tick data.
- **Capacity and real impact calibration** — needs execution data from an
  actual broker.

---

## J. Recommended next steps

1. **Paper-trade the fixed momentum rule.** The execution path now matches the
   backtest, which is what makes the comparison meaningful. Watch tracking error
   and return correlation before P&L.
2. **Buy point-in-time data** if this is going anywhere near real money. It is
   the largest uncontrolled variable remaining.
3. **Do not add features.** The evidence across three parts of this project is
   that every layer of adaptation made results worse. Change that only when a
   specific addition beats the fixed rule on an era reserved *before* it was
   built — which the framework now makes convenient and hard to cheat.
4. **Add CI** running `pytest -q` on every push.
5. **Find a bear market.** Extending the equity history to 2007–2009, or testing
   the same logic on an asset class that had one, is worth more than any further
   refinement on this sample.

---

## The honest summary

The framework is substantially more trustworthy than it was: the timing is
provably consistent, the costs are one model with conservative defaults, the
holdout is protected, the search is an order of magnitude smaller, and the
reporting states its own uncertainty.

**The strategy's evidence did not improve because of any of that** — the
numbers moved a little, mostly because a smaller search generalises better. What
improved is the confidence that the numbers mean what they say. A deflated
Sharpe straddling 0.5, on a sample with essentially no bear market, on a
universe assembled by hindsight, is a reason to paper-trade and keep watching —
not a reason to believe.
