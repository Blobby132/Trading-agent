# Strategy stress test — trying to break it

*Phases 2-10. The strategy was treated as guilty until proven innocent. This is what happened.*

**Verdict in one line: the edge is not established.** The strategy survives the tests of its
internal machinery and fails the tests of its economic content. Details below; the summary table
is §1 and the honest reading is §12.

---

## 1. Scoreboard

| # | Test | Result | What it means |
|---|---|---|---|
| A | Time-period robustness | ⚠️ **Mixed** | 7/10 years positive, but only **4/10 beat buy-and-hold** |
| B | Market regime | ⚠️ **One-sided** | Defensive only: wins in bear, badly lags in bull |
| C | Cross-asset, frozen params | ❌ **FAIL** | 0/4 unseen assets beat buy-and-hold; 2 lost money |
| D | Parameter perturbation | ✅ **PASS** | 16/17 parameters on a plateau, median stability 0.994 |
| E | Null / placebo | ⚠️ **Split** | Timing and direction beat their nulls; bootstrap null inconclusive (§6) |
| F | Cost adversity | ✅ **PASS** | Beats its matched baseline up to ~75 bps/side, 5× the base assumption |
| G | Turnover / capacity | ✅ **PASS** | Max participation 3.1e-05 of daily volume — capacity is a non-issue at $100 |
| H | Survivorship | ⚠️ **Unresolved** | Interface + tests exist; the data to fix it does not |
| I | Look-ahead audit | ✅ **PASS** | 75 poisoning tests, extended to the new code paths |
| J | Backtest ↔ paper parity | ✅ **PASS** | Exact reconciliation, unchanged |
| **—** | **Baselines** | ❌ **FAIL** | **A constant 40% position matches it on Sharpe and beats it on Sortino** |

---

## 2. The result that matters most

Over the agent's own traded window, identical bars, identical costs, same $100 start:

| | final | CAGR | Sharpe | Sortino | maxDD | trades | avg exposure |
|---|---|---|---|---|---|---|---|
| **Agent** (walk-forward, 1,216 evaluations) | $1,029 | 29.1% | **0.916** | 1.106 | −51.3% | 572 | 0.51 |
| `half_invested` — 50% of BTC, forever | $956 | 28.1% | 0.901 | **1.252** | −56.1% | 73 | 0.50 |
| `forty_pct_invested` — 40% of BTC, forever | $661 | 23.0% | 0.901 | **1.252** | **−46.9%** | 101 | 0.40 |
| `momentum_12m` — long if 12-month return > 0 | **$1,446** | 34.0% | 0.853 | 1.005 | −53.2% | 44 | 0.61 |
| `price_above_sma_200` | $923 | 27.6% | 0.784 | 0.796 | −64.4% | 67 | 0.49 |
| `sma_50_200` | $561 | 20.8% | 0.642 | 0.607 | −66.0% | 19 | 0.49 |
| `buy_and_hold` | $2,857 | 44.4% | 0.885 | 1.218 | −83.8% | 1 | 1.00 |

**The agent's Sharpe advantage over holding a constant 40% of Bitcoin and never touching it again
is 0.916 vs 0.901 — 1.6%.** Its Sortino is *worse* than that baseline. At matched average exposure
its drawdown is *worse*. And a one-line 12-month momentum rule makes 41% more money.

Six strategies, adaptive performance weighting, a regime filter, volatility targeting, ATR stops, a
drawdown kill switch, a walk-forward search and 1,216 configuration evaluations produce a
risk-adjusted return indistinguishable from a position size chosen once and left alone.

For scale: the daily cell's own seed spread, measured earlier in this repository, is $938–$1,564
across four seeds. `half_invested` sits at $956 — **inside** the agent's seed noise. The difference
is not merely small, it is smaller than the run-to-run variation of the agent itself.

---

## 3. A. Time-period robustness

Requested eras 2007-2009, 2010-2012 and 2013-2015 **do not exist for this data** — Coinbase BTC
starts 2015-07 and the working equity source starts 2016-09. Stated as missing, not fabricated.

Agent vs buy-and-hold, same bars, by calendar year:

| year | return | benchmark | excess | beat? | Sharpe | maxDD | underwater bars | trades | costs |
|---|---|---|---|---|---|---|---|---|---|
| 2017 (part) | +80.7% | +413.4% | −332.7% | ✗ | 3.19 | −21.1% | 67 | 40 | $0.58 |
| 2018 | −28.9% | −73.4% | **+44.4%** | ✓ | −1.18 | −42.1% | **359** | 184 | $6.37 |
| 2019 | +95.4% | +94.1% | +1.4% | ✓ | 1.90 | −20.9% | 188 | 91 | $2.39 |
| 2020 | +159.4% | +304.6% | −145.2% | ✗ | 2.37 | −18.6% | 163 | 96 | $5.89 |
| 2021 | +4.4% | +59.4% | −55.0% | ✗ | 0.32 | −41.4% | 293 | 97 | $13.74 |
| 2022 | −15.4% | −64.2% | **+48.8%** | ✓ | −1.70 | −19.0% | 277 | 107 | $9.03 |
| 2023 | +24.6% | +155.8% | −131.3% | ✗ | 1.10 | −14.5% | 230 | 104 | $9.54 |
| 2024 | +69.7% | +120.8% | −51.1% | ✗ | 1.41 | −35.2% | 244 | 98 | $23.85 |
| 2025 | −17.5% | −6.3% | −11.2% | ✗ | −0.52 | −31.7% | 223 | 119 | $55.48 |
| 2026 (part) | +2.5% | −11.8% | +14.4% | ✓ | 0.32 | −7.4% | 108 | 26 | $8.81 |

**7 of 10 years positive. 4 of 10 beat buy-and-hold.** Best 2020 (+159%), worst 2018 (−29%). The
two clear wins, 2018 and 2022, are both bear years. 2025 is the worrying one: it lost 17.5% in a
year the benchmark lost only 6.3%, which is the defensive story failing in a mild decline.

359 consecutive underwater bars inside 2018 — most of a year below the previous peak.

---

## 4. B. Market regime

Causal trailing labels (200-bar trend, rolling volatility percentile), never full-sample quantiles.

| dimension | regime | share | strategy | benchmark | excess | Sharpe |
|---|---|---|---|---|---|---|
| trend | **bear** | 30.6% | −38.7% | −80.5% | **+41.8%** | −0.83 |
| trend | **bull** | 51.6% | +828.6% | +2,425.8% | **−1,597.2%** | 1.37 |
| trend | sideways | 11.8% | +19.9% | +84.0% | −64.0% | 0.73 |
| volatility | high_vol | 20.9% | +42.9% | +61.3% | −18.4% | 0.66 |
| volatility | low_vol | 29.0% | +30.8% | +19.0% | **+11.8%** | 0.50 |
| volatility | normal_vol | 47.4% | +402.5% | +600.2% | −197.7% | 1.26 |

**The strategy has exactly one regime where it adds value: bear markets.** It captures roughly a
third of bull-market return and avoids roughly half of bear-market loss. That is a coherent
description of a defensive trend follower — and it is precisely why §2 compares it against a
constant partial position, which provides the same asymmetry for free.

---

## 5. C. Cross-asset validation — the clearest failure

The configuration the BTC walk-forward selected most often (9 of 19 folds) was **frozen** and run
unchanged. No re-selection, no per-asset tuning. Tiers are never pooled.

| tier | asset | Sharpe | final | buy-and-hold | beats B&H? | beats SMA-200? |
|---|---|---|---|---|---|---|
| **trained** | BTC-USD | 1.34 | $53,215 | $27,554 | ✓ | ✓ |
| **validation** | ETH-USD | 1.27 | $22,770 | $18,028 | ✓ | ✗ (SMA made $78,144) |
| **unseen** | SPY | 0.43 | $201 | $352 | ✗ | ✓ |
| **unseen** | QQQ | 0.64 | $412 | $612 | ✗ | ✗ |
| **unseen** | DIA | 0.07 | **$95** | $286 | ✗ | ✗ |
| **unseen** | IWM | 0.10 | **$93** | $238 | ✗ | ✗ |

**0 of 4 unseen assets beat buy-and-hold. Two lost money while their markets roughly tripled.**
Strong on the trained asset, strong on its closest correlate, absent everywhere else — the
signature of a fit, not a mechanism.

### The leverage confound, diagnosed with the hypothesis written first

The frozen config carries `target_vol = 0.8`. On BTC (60-80% realised vol) that is roughly 1×
exposure; on an equity index realising ~18% it demands 4.4×, capped at `max_leverage = 2.0`. So
the unseen assets ran permanently 2× levered.

**Hypothesis, recorded before the run, with both outcomes named:** if the failure is mainly that
calibration artefact, capping gross at 1.0× — changing nothing about the signal — should move the
unseen assets to roughly track buy-and-hold. If the signal itself does not generalise, the losses
will shrink but the underperformance will remain.

| unseen tier | median final | median Sharpe | worst drawdown | beats B&H |
|---|---|---|---|---|
| gross cap 2.0× | $148 | 0.267 | −57.7% | **0 / 4** |
| gross cap 1.0× | $169 | 0.516 | −37.6% | **0 / 4** |

**The second outcome.** Removing leverage halves the drawdowns and doubles Sharpe, and still leaves
nothing beating buy-and-hold. The leverage explains the damage; it does not explain the
underperformance.

The same cap also drops BTC from $53,215 to $12,166 against buy-and-hold's $27,554 — so the trained
asset's outperformance was substantially leverage too.

**Caveats, both of which make the failure worse rather than better.** Nasdaq ETF data is
price-return, so dividends (~1.3-2%/yr) are missing from both sides — but buy-and-hold is invested
100% of the time and the strategy ~50%, so restoring dividends would widen the gap. And 2016-2026
was a strong US equity bull market, which a defensive trend follower will lag by construction; that
mitigates the *underperformance* but not the outright money loss on DIA and IWM.

---

## 6. E. Null and placebo experiments

Four nulls, 20 replications each, every replication re-running the **whole** pipeline — search,
fold loop, cost model. 80 full walk-forwards. What comes back is the distribution of this
machinery's output under each null, which is what the real result has to be compared against.

Observed: Sharpe **0.916**, final **$1,029**.

| null | what it removes | null median Sharpe | null max | observed percentile | p-value | verdict |
|---|---|---|---|---|---|---|
| `shuffled_signal` | *when* the positions occur (block-permuted weights, same exposure and turnover) | 0.455 | 0.863 | 100th | **0.048** | ✅ timing is informative |
| `sign_flipped` | *which way* (block-randomised direction, same magnitude) | −0.223 | 0.604 | 100th | **0.048** | ✅ direction is informative |
| `random_selection` | the optimiser's judgement (top-k picked at random from the same pool) | 0.775 | 0.904 | 100th | **0.048** | ✅ selection adds something |
| `bootstrapped_prices` (raw) | long-horizon trend (circular block bootstrap of the price path) | **1.079** | 1.789 | 35th | 0.667 | ❌ — but see below |

0.048 is the floor for 20 replications (`(0+1)/(20+1)`); nothing smaller is reportable and nothing
smaller is claimed.

### The bootstrap null was confounded, and fixing it reverses the result

Raw, the last row looks damning: the pipeline scores **higher** on trendless synthetic paths than
on real data. That comparison is invalid, and the reason is visible in the data.

A circular block bootstrap preserves the series' unconditional drift while scattering its sustained
bear markets into isolated bad weeks. The synthetic paths are therefore *smoother than reality* and
everything scores better on them — **buy-and-hold's own Sharpe rises from 0.885 on the real path to
a median of 1.51 on the bootstrapped ones.** Comparing absolute Sharpes across that gap measures
how much easier the synthetic world is, not whether the strategy has an edge.

Differencing within each path cancels path difficulty:

| | strategy − buy-and-hold Sharpe, same path |
|---|---|
| bootstrapped paths (n=20) | median **−0.463**, p90 −0.173, best −0.103 — **negative on all 20** |
| **real path** | **+0.031** |
| | **p = 0.048** |

**The strategy adds value over passive exposure only when real trend structure is present, and
destroys about 0.46 of Sharpe when it is absent.** That is exactly what a working trend follower
should do, and it is the strongest positive evidence in this document: the mechanism responds to
something real in the data rather than to noise.

It does not make the magnitude interesting. The real edge is **+0.031 Sharpe** against buy-and-hold
and **+0.015** against a constant 40% position. Statistically distinguishable from the null;
economically negligible.

### What `random_selection` actually says

The optimiser beats random selection from its own candidate pool — 0.916 against a median of 0.775.
But the *best* random draw scored **0.904**, essentially tying the optimiser. So roughly 85% of the
Sharpe is available from the candidate pool with no selection intelligence at all. The search
contributes about 0.14 Sharpe; the strategy family contributes the rest.

---

## 7. D. Parameter perturbation — a genuine pass

Each parameter moved ±20% one at a time, others held. The best neighbour was **not** selected;
this is sensitivity analysis, not optimisation.

| outcome | count | parameters |
|---|---|---|
| plateau (stability ≥ 0.8) | **16 / 17** | |
| SPIKE (collapses nearby) | 1 | `allow_short` |
| a neighbour beats it by >20% | **0** | |

Median stability **0.994**; worst 0.470. The single spike is `allow_short`, a binary flag whose
alternative setting scores 0.434 against 0.923 — turning shorting on makes it worse. That is an
economic finding and the safety-constrained default, not a knob fitted to noise.

**The strategy is not parameter-fragile.** This is the test it passes most clearly, and it is
evidence that whatever the machinery is doing, it is not balanced on a single lucky cell.

Sensitivity is measured over the full traded window with frozen parameters, so it is in-sample with
respect to parameter choice. That is standard for sensitivity analysis and is stated rather than
implied.

---

## 8. F. Cost adversity — where it stops being viable

Frozen configuration, full traded window (in-sample w.r.t. parameter choice — the *shape* is the
finding, not the level).

| bps/side | final | Sharpe | maxDD | vs 40%-constant baseline |
|---|---|---|---|---|
| 0 | $2,447 | 0.981 | −52.3% | +$1,786 |
| 4 (LOW) | $2,264 | 0.961 | −51.7% | +$1,603 |
| **15 (BASE)** | **$1,929** | **0.923** | −53.4% | +$1,269 |
| 35 (HIGH) | $1,460 | 0.859 | −56.0% | +$799 |
| 50 | $1,128 | 0.799 | −58.1% | +$467 |
| 75 | $925 | 0.750 | −62.6% | +$264 |
| 100 | $604 | 0.650 | −68.9% | **−$57** |
| 150 | $294 | 0.481 | −70.7% | −$367 |

**It beats its matched baseline up to about 75 bps per side — five times the base assumption — and
never loses money outright even at 150 bps.** Genuine robustness, and the clearest counterweight to
the negative findings. Note this is the *daily* configuration; the hourly one was previously shown
to collapse to 21% of gross profit under HIGH costs.

---

## 9. G. Turnover and capacity

| | |
|---|---|
| trades | 572 (62.6/yr) |
| turnover | 18.6× equity/yr |
| average trade notional | $253 |
| largest trade notional | $3,942 |
| **participation of daily dollar volume** | median 2.6e-07, **max 3.1e-05** |
| costs paid | $217 (217% of the $100 stake) |
| average gross exposure | 0.51 |
| time in market | 49.8% |

At a $100 stake market impact is unmeasurable — the largest trade is three thousandths of one
percent of BTC's daily volume. **Capacity is not a constraint here and the returns do not depend on
unrealistic volume.** The cost *rate* matters enormously (§8); the cost of *size* does not.

---

## 10. H. Survivorship bias — unresolved, and stated as such

* **Crypto: no exposure.** One asset, still trading.
* **Stocks: material and unquantified.** `US_LARGE_CAP` is today's list applied to all history.

What now exists: `MembershipProvider` / `StaticMembership` / `PointInTimeMembership` /
`apply_membership`, a `membership` field in the research ledger so a biased result and a clean one
are no longer indistinguishable in the record, and three tests that fail if a future constituent
leaks — a name entering the index in 2020 must be untradeable in 2018, a delisted name must stop
being tradeable, and `StaticMembership.describe()` must say "NOT point-in-time" in words.

**This is not a fix.** It is an interface plus a guard. Fixing it needs point-in-time index
membership with entry/exit dates *and* delisted-name price history — Norgate, Sharadar or CRSP.
None is reachable from this environment. **Survivorship bias has not been solved and is not
claimed to be.**

---

## 11. I & J. Look-ahead and parity

Unchanged and still passing. The look-ahead audit works by **future poisoning** — replace every bar
after a cut with garbage, assert the past is bit-identical — with negative controls that fail if
the poisoning stops biting. It now also covers the falsification machinery: block permutation,
sign flipping, the bootstrap path, and the random-selection null.

Backtest ↔ paper parity reconciles exactly under every cost scenario, through the single
`execution.plan_rebalance` both paths call.

**425 tests pass, 2 skipped.** No test was skipped due to an error.

---

## 12. Phases 4-7: selection hygiene, statistical honesty, and the gate

**Ledger (Phase 4).** The research ledger recorded 1 experiment and 0 used for selection while the
interval study had evaluated 15,808 configurations. It is now wired into every search path and
gained a `membership` field. Every claim of the form "corrected for N trials" was previously
correct only because a human typed N into a report.

**Provenance (Phase 3).** `provenance.py` labels every result IN-SAMPLE / OUT-OF-SAMPLE / HOLDOUT /
UNSEEN-ASSET / UNSEEN-PERIOD, requires the configuration count rather than defaulting it to 1, and
**raises** if asked to average across kinds.

**The gate (Phase 7).** Default-deny across ten dimensions; a check that was never run counts as
failed. Run on the real strategy with the real evidence:

| check | status | detail |
|---|---|---|
| oos_performance | ✅ PASS | sharpe 0.92, drawdown −51.3%, 572 trades |
| **beats_matched_baseline** | ❌ **FAIL** | sharpe 0.916 vs 0.901 (+0.015); **sortino 1.106 vs 1.252** |
| parameter_stability | ✅ PASS | median stability 0.994 |
| cost_robustness | ✅ PASS | beats the matched baseline to ~75 bps/side |
| **regime_robustness** | ❌ **FAIL** | positive excess in **1/3** trend regimes (bear only) |
| **cross_asset** | ❌ **FAIL** | **0/4** unseen assets beat buy-and-hold |
| **statistical_evidence** | ❌ **FAIL** | **deflated Sharpe 0.288** (a coin scores 0.50) |
| capacity | ✅ PASS | max participation 3.1e-05 of daily volume |
| execution_realism | ✅ PASS | parity tests pass |
| holdout_discipline | ⚠️ advisory FAIL | the reserved era has been evaluated against |

### **VERDICT: NOT ACCEPTED — 4 blocking checks.**

---

## 13. The honest reading

**Does the strategy still look promising after trying to break it? No — but it is not nothing.**

What genuinely survived:

* It is **not parameter-fragile** (16/17 on a plateau, median stability 0.994).
* It is **not cost-fragile** at daily frequency (beats its matched baseline to ~75 bps/side).
* It is **not a capacity fantasy** (largest trade is 3/100,000ths of daily volume).
* It **beats three of four nulls outright**, and the fourth once the confound is removed. Timing,
  direction and selection each carry real information, and the drift-controlled test shows the
  mechanism responds to genuine trend structure.
* It is **genuinely defensive**: +41.8% excess in bear regimes, −52% of the benchmark's drawdown.

What failed, and why it matters more:

* **A constant 40% position matches it.** Sharpe 0.916 vs 0.901, and the baseline's Sortino is
  *better*. The entire apparatus buys 1.6% of Sharpe over a number chosen once.
* **It works in one regime.** Bear markets. In bull markets it captures a third of the move.
* **It does not transfer.** 0/4 unseen assets beat buy-and-hold; two lost money outright.
* **Deflated Sharpe 0.288.** After 1,216 configurations, the probability the true Sharpe exceeds
  zero is below a coin flip.
* **The edge is 0.031 Sharpe.** Real, measurable, and too small to act on.

The reconciliation: the mechanism is probably real and probably tiny. The nulls say something
genuine is being detected. The baselines say what is detected is worth about as much as picking a
position size and going away.

## 14. Biggest remaining source of uncertainty

**One asset, one era, containing two crypto bull markets.** Every positive finding rests on
BTC-USD 2015-2026. The cross-asset test is the attempt to escape that, and it failed. There is no
prolonged slow bear market in any dataset reachable from here, and that is the regime a trend
follower is most likely to fail in.

Second: **survivorship bias in the stock universe is unresolved** and cannot be resolved without
vendor data.

## 15. What should be tested next

1. **A drift-controlled null on the unseen assets.** The single highest-value experiment. The
   cross-asset failure and the drift-controlled null point in opposite directions; running the
   latter on SPY/QQQ/IWM/DIA would say whether the mechanism is genuinely absent there or merely
   swamped by a strong bull market. One run, and it resolves the central contradiction.
2. **More seeds on the drift-controlled comparison.** p = 0.048 is the floor at 20 replications;
   100 would allow p ≈ 0.01 and make the strongest positive finding much harder to dismiss.
3. **A volatility-scale-free frozen configuration for cross-asset work** — freeze the *signal* and
   let `target_vol` be set by the asset's own realised volatility. Stated as a hypothesis to test,
   not an improvement to adopt.

## 16. What changes should be accepted

**None to the strategy.** No parameter was tuned, no configuration was adopted, and the live paper
account is untouched. Every change in this work was to the *measurement apparatus*: the nulls, the
baselines, the cross-asset harness, the ledger wiring, the provenance labels, the gate, and the
tests that keep them honest.

That is the correct outcome. The system can now say "this strategy does not appear to have a
reliable edge" — and on the evidence collected here, that is what it says.
