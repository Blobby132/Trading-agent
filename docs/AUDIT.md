# Repository audit

Full-lifecycle review of the research framework, conducted before any changes were made.
Scope: every module in `tradingagent/`, the test suite, notebooks, packaging and docs.

Severity: **CRITICAL** (results are wrong or unachievable) · **HIGH** (materially biases results)
· **MEDIUM** (misleading or fragile) · **LOW** (hygiene).

---

## Lifecycle traced

```
data.py ──► features.py / indicators.py ──► strategies.py / cross_section.py ──► agent.py
   │                                                                               │
   │                                                              risk.py / risk_learner.py
   │                                                                               │
   └──► universe.py (Panel) ──────────────────────────────────────────► target weights
                                                                                   │
                                                        engine.py (fills, costs, financing)
                                                                                   │
              metrics.py ◄── optimize.py / xs_optimize.py (walk-forward) ◄──────────┘
                                        │
                     holdout.py ────────┘                paper.py (forward, live-vs-backtest)
```

Two independent execution paths exist: `engine.py` for everything backtested, and `paper.py`
for forward trading. **They do not implement the same causal model.** That is the headline
finding.

---

## 1. Architecture strengths

- **The timing convention is correct in the backtest.** `engine.run` shifts target weights by
  one bar and fills at the next bar's open. Decisions made on bar `T`'s close cannot transact
  at that close.
- **Path-dependent effects are modelled honestly.** Compounding, intrabar stops and the
  drawdown kill switch are simulated in a bar loop rather than vectorised away.
- **Walk-forward with purge and embargo.** `xs_optimize` purges training rows whose forward
  return resolves after the training window, then embargoes further bars.
- **A real look-ahead audit already exists.** `tests/test_no_lookahead.py` poisons the future
  rather than truncating it, and carries negative controls that prove it can fail.
- **Costs compound against the account** rather than being subtracted at the end.
- **A changing universe is handled.** NaN prices mean "not tradeable"; stranded positions are
  liquidated at the last printed price.
- **Deflated Sharpe and block-bootstrap** are already reported, and reported as intervals.

## 2. Architecture weaknesses

| # | Issue | Severity |
|---|---|---|
| A1 | **Two execution implementations.** `engine.py` and `paper.py` duplicate order sizing, dust filtering and cost charging in separate code. They have already drifted (see §7). | **CRITICAL** |
| A2 | No shared cost/execution abstraction — assumptions are literals scattered across six call sites. | **HIGH** |
| A3 | `holdout.py` is not wired into either optimizer. Nothing prevents passing the reserved era straight into `walk_forward`. | **HIGH** |
| A4 | No research ledger. Experiments are not recorded, so repeated searching is invisible. | **HIGH** |
| A5 | `engine.run` recomputes ATR per call; walk-forward calls it thousands of times on overlapping windows. | MEDIUM |
| A6 | No CI configuration. The suite is only run manually. | LOW |

## 3. Look-ahead risks

| # | Issue | Severity | Status |
|---|---|---|---|
| L1 | **`paper.plan_orders` sizes at the signal bar's close and `apply_orders` fills at that same close.** The docstring claims "executed at the next open"; the code does not. Paper trading therefore transacts on information at the instant it is generated. | **CRITICAL** | open |
| L2 | `engine` sizes `target_units = weight × equity / open[T+1]`, i.e. the fill price is used to compute order size. Defensible for *notional* (fractional-share) orders, where the broker fills a dollar amount at whatever the open is — but it is an assumption, undocumented, and impossible for share-count orders. | MEDIUM | open |
| L3 | `Panel.min_history(bars)` selects universe members using total bars across the **whole sample**, including the future. | HIGH | known, documented; `require_history_before` exists but is not the default |
| L4 | Cross-sectional standardisation is per-date (uses every name's value on the same date). Not look-ahead, but it assumes the cross-section can be computed between the close and the next open. | LOW | acceptable for daily bars, undocumented |
| L5 | Rolling stats, forward-return targets, trailing-stop updates, benchmark curves. | — | **clean** — covered by poisoning tests |

## 4. Survivorship-bias risks

| # | Issue | Severity |
|---|---|---|
| S1 | `US_LARGE_CAP*` are hand-written lists of names liquid **today**. Not point-in-time. | **HIGH** |
| S2 | There is no interface for historical index membership — the concept does not exist in the code. | HIGH |
| S3 | The Nasdaq data source serves **no** delisted tickers at all, so any run on that source is structurally survivor-only. | HIGH |
| S4 | Measured, not assumed: `survivorship.py` shows failures cost the equal-weight benchmark ≈2× what they cost a top-decile momentum strategy. The bias is real but flatters the **benchmark** more. | — |

## 5. Selection / optimization-bias risks

| # | Issue | Severity |
|---|---|---|
| O1 | **Single-asset search space is 20 parameters, 3.27×10⁹ combinations**, sampled 150× per fold across ~19 folds. Enormous relative to 3,333 bars. | **CRITICAL** |
| O2 | The same historical folds are re-evaluated on every seed and every re-run, with no record. | HIGH |
| O3 | Seed dependence is measurable (final equity ranged $706–$1,390 across 5 seeds) but is not part of the acceptance criteria. | HIGH |
| O4 | No parameter-stability testing. A configuration selected at lookback=50 is never checked at 45/55. | HIGH |
| O5 | Objective `target_growth` embeds a drawdown penalty with hand-chosen constants (8.0, 0.25) that were never validated. | MEDIUM |
| O6 | Deflated Sharpe is computed but is **not** an acceptance gate — results below 0.5 are still reported as headline numbers. | MEDIUM |

## 6. Transaction-cost realism

| # | Issue | Severity |
|---|---|---|
| C1 | **Six different cost assumptions across the repo**: engine default 10+5 bps; `buy_and_hold_equity` 10 bps; `paper.apply_orders` 8 bps combined; `paper.divergence_report` 5+3 bps; `equal_weight_benchmark` **1+1 bps**; CLI 10+5 bps. | **HIGH** |
| C2 | The benchmark is charged 2 bps while strategies pay 8–16 bps. Unexplained, and it makes the benchmark harder to beat — conservative for the strategy, but arbitrary. | HIGH |
| C3 | **No market impact.** Cost is linear in notional regardless of order size; a $100 order and a $100m order pay the same rate. | MEDIUM (LOW at $100 account size) |
| C4 | **No bid/ask spread model.** Slippage is a flat bps charge, not a half-spread crossed on entry and exit. | MEDIUM |
| C5 | Slippage is charged as a **cash cost, not an adverse fill price**, so the recorded trade `price` is the raw open and understates what was actually paid. P&L is right; the trade log is not. | MEDIUM |
| C6 | No LOW/BASE/HIGH scenario capability — cost sensitivity cannot be swept. | HIGH |

## 7. Execution-model inconsistencies (backtest vs paper)

| # | Divergence | engine.py | paper.py | Severity |
|---|---|---|---|---|
| E1 | Fill timing | next bar's **open** | **same bar's close** | **CRITICAL** |
| E2 | Sizing price | the fill price (open) | the prior close | **HIGH** |
| E3 | Cost rate | `fee_bps + slippage_bps`, configurable | hard-coded `cost_bps=8.0` | HIGH |
| E4 | Dust filter | position-relative, in engine | position-relative, **re-implemented** in paper | MEDIUM |
| E5 | Stops / kill switch / lockout | modelled | absent | MEDIUM |
| E6 | Financing / borrow | modelled | absent | LOW (long-only unlevered) |

`divergence_report` compares paper against a backtest, but because E1–E3 differ, part of any
measured tracking error is the framework disagreeing with itself rather than reality
disagreeing with the model.

## 8. Risk-management weaknesses

| # | Issue | Severity |
|---|---|---|
| R1 | **Kelly estimator is statistically unsound at the sample sizes used.** `mu/var` from as few as `lookback//2` = 30 daily observations, capped at 4×. The standard error of that ratio at n=30 is larger than the estimate. | **HIGH** (mitigated: disabled by default) |
| R2 | Kelly has no shrinkage, no minimum sample size, and no standard-error guard. | HIGH |
| R3 | The drawdown kill switch is a cliff (flat/not-flat) and resets its high-water mark on expiry — better than latching, but still discontinuous. `risk_learner.drawdown_scale` is the smooth version and is not the default. | MEDIUM |
| R4 | `max_leverage` defaults to 2.0 in `RiskConfig`/`ExecutionConfig` — leverage on by default in a research framework. | MEDIUM |
| R5 | No position-level or portfolio-level exposure limit beyond gross leverage (no sector/correlation cap). | LOW |

## 9. Statistical-validation weaknesses

| # | Issue | Severity |
|---|---|---|
| V1 | No confidence intervals on headline statistics. Sharpe is reported as a point estimate. | HIGH |
| V2 | No regime decomposition. Results are reported over one blended period. | HIGH |
| V3 | No trade-level statistics: expectancy, average/largest win and loss, win rate by trade (only by bar). | MEDIUM |
| V4 | No drawdown *duration*, no downside volatility reported (Sortino uses it internally). | MEDIUM |
| V5 | Gross vs net return is not separated; fees and slippage are not reported apart. | MEDIUM |
| V6 | No acceptance criteria defined anywhere — nothing states what would make a strategy fail. | HIGH |

## 10. Testing gaps

| # | Gap | Severity |
|---|---|---|
| T1 | **No backtest/paper parity test.** The two execution paths are never compared. | **CRITICAL** |
| T2 | No deterministic fixed-OHLC fixture asserting exact fills, fees and slippage. | HIGH |
| T3 | No test that the holdout cannot be consumed by an optimizer. | HIGH |
| T4 | No reproducibility test (same seed ⇒ identical result). | MEDIUM |
| T5 | No test of gap-through-the-open handling in paper. | MEDIUM |
| T6 | Engine tests use synthetic random data; few assert exact arithmetic. | MEDIUM |

## 11. Performance bottlenecks

| # | Issue | Severity |
|---|---|---|
| P1 | ATR is recomputed inside `engine.run` on every call; walk-forward calls it thousands of times over overlapping windows. | MEDIUM |
| P2 | `feature_panel` is recomputed per candidate in some paths rather than cached. | MEDIUM |
| P3 | `Panel.to_frames()` rebuilds per-symbol frames on every call, including inside fold loops. | MEDIUM |
| P4 | The per-bar Python loop remains the floor on engine speed (already reduced ~2.4× by visiting only changed assets). | LOW |

## 12. Bugs that can materially affect results

| # | Bug | Severity | Status |
|---|---|---|---|
| B1 | `paper.plan_orders` docstring states next-open execution; implementation fills at the signal close. Docs and code disagree, and the code is wrong. | **CRITICAL** | to fix |
| B2 | `equal_weight_benchmark` silently charges 1+1 bps against strategies paying 8–16. | HIGH | to fix |
| B3 | `paper.apply_orders` average-price tracking only updates on `shares > 0`; a partial **sell** leaves `avg_price` stale, so reported per-position P&L drifts. | MEDIUM | to fix |
| B4 | `ExecutionConfig.record_trades=False` makes `total_costs` fall back to `costs.sum().sum()`, which includes financing — so "costs" means different things depending on a performance flag. | MEDIUM | to fix |
| B5 | No guard against `min_trade_frac ≥ 1.0`, which silently disables all rebalancing. | LOW | to fix |

---

## Ranked remediation order

1. **CRITICAL** — one canonical execution model shared by backtest and paper (A1, L1, E1–E3, B1, T1).
2. **CRITICAL** — deterministic parity tests that fail loudly on any timing regression (T1, T2).
3. **HIGH** — one canonical cost model with LOW/BASE/HIGH scenarios (C1, C2, C6, B2).
4. **HIGH** — holdout protection wired into the optimizers (A3, T3).
5. **HIGH** — research ledger (A4, O2).
6. **HIGH** — parameter-stability / perturbation testing (O4).
7. **HIGH** — shrink the single-asset search space (O1).
8. **HIGH** — Kelly guard + statistical honesty (R1, R2).
9. **HIGH** — regime decomposition and expanded statistics with uncertainty (V1–V5).
10. **MEDIUM** — robustness battery, point-in-time universe interface, performance.
