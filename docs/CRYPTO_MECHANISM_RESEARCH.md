# Why the mechanism works on crypto — and one improvement that failed

**Classification: `MECHANISM IDENTIFIED`.** The structural precondition is found and it is
categorical. **The one improvement tested was `IMPROVEMENT REJECTED`** — it passed on BTC and
failed independent validation on ETH.

The production configuration is unchanged. Paper/live behaviour is untouched.

---

## 1. Phase 1 — the control, reproduced

`CRYPTO_CONTROL_CURRENT` ≡ `BTC_FLOOR_050_CONTROL`, unchanged. Reproduction was verified before any
research, against `docs/INDEPENDENT_EDGE_VALIDATION.md`, on BTC, ETH and SPY. **All three reproduce
exactly.**

| | final equity | CAGR | Sharpe | Sortino | maxDD | trades | turnover | avg exposure | time in mkt | exp p10 / p50 / p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| BTC-USD | $126,200 | 89.7% | 1.412 | 1.827 | −57.3% | 611 | 13.4× | 0.786 | 76.4% | 0.00 / 0.77 / 1.65 |
| ETH-USD | $22,497 | 69.2% | 1.219 | 1.549 | −63.5% | 570 | 12.0× | 0.526 | 71.2% | 0.00 / 0.51 / 1.13 |
| SPY | $186 | 6.4% | 0.370 | 0.410 | −51.4% | 77 | 5.1× | **1.620** | 84.5% | 0.00 / **1.93** / 2.02 |

Regime performance of the control (excess vs buy-and-hold):

| | bear | bull | sideways | sideways share |
|---|---|---|---|---|
| BTC-USD | +0.627 | +633.96 | −0.625 | **9.7%** |
| ETH-USD | +0.542 | −1750.20 | −0.441 | **9.9%** |
| SPY | +0.242 | +3.502 | **−1.184** | **31.9%** |

SPY's median exposure is 1.93 — pinned near the leverage cap — and it spends 32% of its life in the
sideways regime where the strategy lost 69%. Crypto spends ~10%.

---

## 2. Phase 2 — what actually separates crypto from equities

Six assets, descriptive statistics only, no strategy involved.

### The finding: momentum autocorrelation flips sign

Correlation between one *h*-bar return and the next:

| asset | 1-month | 3-month | 6-month | 12-month |
|---|---|---|---|---|
| **BTC-USD** | **+0.075** | **+0.147** | +0.079 | −0.079 |
| **ETH-USD** | **+0.177** | **+0.301** | +0.009 | +0.033 |
| SPY | −0.156 | −0.224 | −0.036 | −0.392 |
| QQQ | −0.099 | −0.146 | +0.048 | −0.330 |
| DIA | −0.177 | −0.273 | −0.137 | −0.494 |
| IWM | −0.084 | −0.167 | −0.030 | −0.424 |

**Momentum continues in crypto and mean-reverts in every equity index tested.** That is precisely
the precondition a trend follower requires, and it is a sign difference, not a magnitude
difference.

### Four variables where the ranges are *disjoint*

| variable | crypto | equity | |
|---|---|---|---|
| 1-month momentum autocorrelation | [+0.075, +0.177] | [−0.177, −0.084] | **disjoint** |
| 3-month momentum autocorrelation | [+0.147, +0.301] | [−0.273, −0.146] | **disjoint** |
| share of bars in sideways regime | [9.7%, 9.9%] | [16.5%, 43.5%] | **disjoint** |
| share of bars in bear regime | [25.0%, 31.4%] | [4.3%, 12.6%] | **disjoint** |

Rank correlation with the strategy's matched-exposure Sharpe edge across the six assets — with six
points these are descriptive, not inferential:

| variable | ρ |
|---|---|
| sideways share | **−0.943** |
| 3-month momentum autocorrelation | **+0.886** |
| 1-month autocorrelation / bear share / annualised vol | +0.771 |
| median drawdown depth | −0.771 |

### Other structure, for completeness

| | BTC | ETH | SPY | QQQ | DIA | IWM |
|---|---|---|---|---|---|---|
| annualised volatility | 67% | 93% | 18% | 23% | 18% | 23% |
| volatility clustering (AC1 of \|r\|) | 0.21 | 0.23 | 0.36 | 0.27 | 0.40 | 0.26 |
| median drawdown depth | −8.3% | −10.2% | −1.2% | −1.8% | −1.0% | −2.3% |
| worst drawdown | −83.8% | −94.0% | −34.1% | −35.6% | −37.1% | −42.3% |
| median trend run (90-bar sign) | 5 | 3 | 4 | 4 | 5 | 3 |

**Volatility clustering is *higher* in equities, not crypto** — so the "volatility clustering makes
vol targeting work" story does not separate the asset classes. Trend-run lengths are also similar.
The separation is momentum continuation and regime occupancy, not clustering or run length.

### The 24/7 hypothesis — tested, and refuted

| | full history | weekdays only | weekend vol ÷ weekday vol |
|---|---|---|---|
| BTC-USD | Sharpe 1.412 | **1.191** | 0.71 |
| ETH-USD | Sharpe 1.219 | **0.989** | 0.79 |

Removing every weekend bar leaves both cryptos far above every equity index (0.09–0.70). Weekend
bars are *lower* volatility than weekdays. **Continuous trading is not the explanation.**

---

## 3. Phase 3 — why the BTC configuration works on ETH

ETH matches BTC on all four disjoint variables, and exceeds it on the key one (3-month
autocorrelation +0.301 against +0.147). ETH is a harder, more volatile instrument (93% vs 67%
annualised, worst drawdown −94.0% vs −83.8%) with *more* momentum continuation.

**The smallest set of shared characteristics absent from equities: positive short-to-medium-horizon
momentum autocorrelation, plus a low share of sideways regime.** The other measured differences
(volatility level, drawdown depth, bear-market share) follow from or co-move with those two.

---

## 4. Phase 4 — component attribution, and a correction to earlier phases

Frozen configuration, ablated one component at a time, on three assets. Change in final equity:

| ablation | BTC | ETH | SPY | reading |
|---|---|---|---|---|
| **− trend floor** | **−57.8%** | **+1.2%** | +7.8% | **BTC-specific** |
| − volatility targeting | −87.1% | −26.7% | −11.3% | matters everywhere, most on crypto |
| **− drawdown kill switch** | **+27.4%** | **+26.2%** | −4.6% | helps crypto, hurts equity |
| + adaptive blending | +10.6% | +29.2% | +5.2% | current setting is not optimal anywhere |
| + signal smoothing (1→5) | −54.2% | +5.7% | +16.6% | current setting right for BTC only |
| + regime filter | −45.7% | −16.3% | +7.7% | correctly off for crypto |
| + ATR stop | −71.5% | −63.9% | −4.7% | correctly off |
| full 6-model ensemble | −22.6% | −11.3% | −0.9% | 3-model subset is better |

### ⚠️ This corrects a conclusion from two earlier phases

**The trend floor — accepted in the edge-improvement phase and carried into the control — does
essentially nothing on ETH.** Removing it costs BTC 57.8% of final equity and *gains* ETH 1.2% and
+0.054 Sharpe.

Earlier work described the BTC→ETH result as validating "the mechanism", with the floor as the
headline improvement. That conflated two things. What travels is **trend-following plus volatility
targeting** (−87.1% / −26.7% / −11.3% when removed). The floor is a BTC-specific addition sitting on
top of a mechanism that does not need it.

---

## 5. Phase 5 — hypotheses

**H1 — the travelling mechanism is volatility targeting, not the trend floor.**
*Evidence:* the ablation above. *Prediction:* ETH's edge survives without the floor; BTC's does not.
*Falsification:* ETH degrades materially when the floor is removed.
*Result:* **SUPPORTED.** ETH +1.2% equity and +0.054 Sharpe without the floor; BTC −57.8%. The floor
is not the travelling component.

**H2 — the drawdown kill switch destroys value on crypto.**
*Evidence:* removing it gains +27.4% BTC, +26.2% ETH, costs −4.6% SPY.
*Prediction:* removal improves both cryptos out of sample without worsening drawdown.
*Falsification:* drawdown worsens, or the gain fails to replicate on ETH.
*Result:* **REJECTED at validation** — see §7.

**H3 — positive momentum autocorrelation is the structural precondition.**
*Evidence:* the sign-disjoint table in §2. *Prediction:* assets with positive autocorrelation
support the strategy; assets with negative do not.
*Falsification:* an asset with positive autocorrelation on which the strategy loses to matched
exposure, or vice versa.
*Result:* **SUPPORTED across all six assets**, with no exceptions — but six assets, two of them
crypto, one era.

**H4 — volatility clustering enables the vol-targeting contribution.** *REFUTED.* Clustering is
*higher* in equities (0.27–0.40) than crypto (0.21–0.23), so it cannot explain the split.

**H5 — continuous 24/7 trading explains the difference.** *REFUTED.* §2.

---

## 6. Phases 6–7 — the improvement experiment

Only H2 produced a candidate. It is a **simplification** — it removes a component and a degree of
freedom — which is the direction the complexity penalty prefers.

**Frozen candidate:** control + `max_drawdown_stop = 0.0`, nothing else changed.

### BTC selection arm (walk-forward, 4 seeds, paired)

| variant | median final | paired equity | wins | t | ΔSharpe | ΔSortino | ΔmaxDD |
|---|---|---|---|---|---|---|---|
| M0 control | $2,475 | — | — | — | — | — | — |
| **M1 no kill switch** | **$2,631** | **+8.23%** | **4/4** | **+3.87** | +0.017 | +0.024 | −0.003 |
| M2 no floor | $1,225 | −46.45% | 0/4 | −6.65 | −0.125 | −0.313 | −0.004 |
| M3 no floor + no kill | $1,130 | −49.17% | 0/4 | −6.99 | −0.144 | −0.338 | −0.037 |

M1 looked good: more money, better Sharpe and Sortino, drawdown unchanged, on every seed.

### The frozen vector on all six assets — a clean asset-class split

| asset | Δ equity | Δ Sharpe | Δ Sortino | Δ maxDD |
|---|---|---|---|---|
| BTC-USD | +27.4% | +0.034 | +0.066 | −0.001 |
| ETH-USD | +26.2% | +0.034 | +0.069 | **+0.092** |
| SPY | −4.6% | −0.018 | −0.019 | −0.028 |
| QQQ | −7.5% | −0.026 | −0.030 | −0.032 |
| DIA | −11.7% | −0.049 | −0.056 | −0.057 |
| IWM | −11.3% | −0.038 | −0.046 | −0.040 |

Helps both cryptos on every metric, hurts all four equities on every metric.

---

## 7. Phase 8 — independent validation, and the rejection

The candidate was frozen before ETH was run and was not modified afterwards.

| | median final | paired equity | wins | t | ΔSharpe | ΔSortino | ΔmaxDD |
|---|---|---|---|---|---|---|---|
| BTC (selection) | $2,631 vs $2,475 | +8.23% | **4/4** | **+3.87** | +0.017 | +0.024 | −0.003 |
| **ETH (validation)** | **$406 vs $420** | +8.60% | **2/4** | **+0.56** | **−0.001** | +0.011 | **−0.035** |

**Rejected.** On ETH's own walk-forward the median final equity is *lower*, the drawdown is 3.5
percentage points *worse*, Sharpe is flat, and the paired t-statistic is +0.56 against +3.87 on BTC.
A candidate whose improvement is smaller than seed noise on the validation asset has not
demonstrated anything.

Downstream tests (chronological thirds, LOW/BASE/HIGH costs, exposure-matched baselines, nulls) were
**not** run on the candidate. It failed the cheap validation gate, and running expensive tests on a
rejected candidate is budget spent dressing up a negative result.

### The discrepancy is itself the finding

The *frozen vector* test gave ETH +26.2%; ETH's *own walk-forward* gives +8.6% and no significance.
The difference is that in the second case ETH selects its own parameters from its own history.
**When the strategy can choose parameters on the asset in front of it, the kill-switch benefit
disappears** — so removal was compensating for a BTC-specific parameter mismatch, not fixing
something real.

### A mechanism I proposed, and its own diagnostic refuted

I hypothesised that crypto drawdowns are deep but recoveries fast, so a kill switch standing aside
after −35% sells the recovery. That predicts a positive *recovery premium*. Measured:

| | share of bars >20% underwater | forward-60 return when deep | unconditional | premium |
|---|---|---|---|---|
| BTC-USD | 62.5% | +0.056 | +0.084 | **−0.028** |
| ETH-USD | 85.6% | +0.064 | +0.083 | **−0.018** |
| SPY | 3.0% | +0.088 | +0.031 | **+0.057** |

The premium is **negative in crypto and positive in equities** — the opposite of the story. On ETH
the bars where the engine flattened against the signal returned +0.0117 against +0.0274
unconditional, so those exits were justified on average. The diagnostic is also weak by
construction for crypto: BTC is >20% underwater on 62% of bars and ETH on 86%, so the conditioning
event is not rare there and the comparison is close to degenerate.

**The kill-switch effect is empirically consistent and mechanistically unexplained.** It is recorded
as such.

---

## 8. Phase 9 — complexity

| | control | candidate |
|---|---|---|
| free parameters | 22 | **21** (one removed) |
| components | trend panel, blend, vol targeting, floor, kill switch | same minus kill switch |
| BTC performance | baseline | +8.2% equity, t = +3.87 |
| ETH performance | baseline | +8.6% equity, **t = +0.56**, drawdown 3.5pp worse |
| robustness | — | **fails independent validation** |

The candidate *reduces* complexity, which normally argues for adoption at equal performance. It is
still rejected, because the performance claim does not survive the validation asset. Simplicity
is a tie-breaker, not a substitute for evidence.

---

## 9. Experiment ledger

Recorded in `results/research_ledger.jsonl`: **11 experiments, 7 used for selection, 23,104
effective trials.**

| experiment | runs | selection? |
|---|---|---|
| control reproduction (BTC/ETH/SPY) | 3 | no — verification |
| market-structure comparison | 6 assets | no — descriptive |
| 24/7 weekend test | 2 | no |
| component ablation | 10 × 3 = 30 | no — diagnostic |
| BTC mechanism walk-forward | 4 × 4 = 16 | **yes** |
| ETH independent validation | 2 × 4 = 8 | no — validated only |
| frozen candidate, all assets | 6 × 2 = 12 | no |
| recovery-structure diagnostic | 3 | no |
| structural screen | 6 | no |
| **candidate accepted** | | **0** |

Nothing was deleted. The rejected candidate has its own ledger entry recording why.

---

## 10. New production code

`tradingagent/market_structure.py` — a **screen, not a signal**. It answers "is this asset a
candidate at all" before any backtest, from the four structural statistics above. Run on the six
assets:

| asset | score | ac_1m | ac_3m | sideways | bear | verdict |
|---|---|---|---|---|---|---|
| BTC-USD | **4/4** | +0.075 | +0.147 | 9.7% | 25.0% | candidate |
| ETH-USD | **4/4** | +0.177 | +0.301 | 9.9% | 31.4% | candidate |
| SPY | 0/4 | −0.156 | −0.224 | 31.9% | 5.8% | unsuitable |
| QQQ | 0/4 | −0.099 | −0.146 | 16.5% | 8.2% | unsuitable |
| DIA | 0/4 | −0.177 | −0.273 | 42.6% | 4.3% | unsuitable |
| IWM | 0/4 | −0.084 | −0.167 | 43.5% | 12.6% | unsuitable |

Thresholds on the autocorrelations are **zero** — a sign change is a claim about the market's
character, whereas a midpoint between six observed points would be a threshold fitted to six points.

11 tests, including one that pins the standard error to *independent windows* rather than bars (a
decade of daily data is ~56 independent 3-month windows, not 3,650 observations; using the bar
count would understate the error eightfold). Writing those tests found a real bug: a constant-growth
series returned a confident **−0.364** autocorrelation computed from floating-point noise at 1e-16,
which would have labelled a low-variance asset "mean-reverting, unsuitable". Now guarded.

---

## 11. Final classification

### `MECHANISM IDENTIFIED` — and the one improvement tested was `IMPROVEMENT REJECTED`

**1. What the evidence says the edge is.** A trend-following edge that requires momentum to
continue. Momentum autocorrelation is positive at 1 and 3 months in both cryptos and negative in all
four equity indices — a sign difference with disjoint ranges — and crypto spends ~10% of its life in
the sideways regime against 16–44% for equities. The strategy earns its return where trends persist
and bleeds where they do not.

**2. What it does not explain.** Why crypto has that property. Volatility clustering is *higher* in
equities, so it is not clustering; removing weekends barely dents the result, so it is not 24/7
trading; trend-run lengths are similar, so it is not run length. The measurement is solid and the
cause underneath it is not established.

**3. Which components are responsible.** Volatility targeting is the largest single contributor and
travels across all three assets (−87.1% / −26.7% / −11.3% when removed). The trend floor is
**BTC-specific** and contributes nothing on ETH. The kill switch costs money on both cryptos. The
ATR stop and regime filter are correctly disabled.

**4. Did the improvement survive independent validation?** **No.** BTC +8.23% at t = +3.87; ETH
+8.60% at t = +0.56 with lower median equity and a 3.5pp worse drawdown.

**5. What failed.** The kill-switch removal (failed ETH validation); the volatility-clustering
hypothesis (clustering is higher in equities); the 24/7 hypothesis (edge survives weekend removal);
and my proposed recovery-premium mechanism (premium is negative in crypto, positive in equities).

**6. What remains uncertain.** Whether "crypto" or "high momentum-continuation assets" is the right
domain; why crypto has positive momentum autocorrelation at all; whether the kill-switch effect is
real or an artifact of BTC-fitted parameters; and everything downstream of two crypto assets in one
era.

**7. Next most informative research question.**

> **Does the structural screen predict success on a non-crypto asset with positive momentum
> autocorrelation?**

The screen now makes a falsifiable, pre-registerable prediction rather than a description. Find
instruments that score 3–4/4 and are *not* crypto — some commodities, EM currencies and volatility
products plausibly qualify — score them **before** backtesting, then run the frozen configuration
with the 1.0× gross cap and nothing else. If the screen predicts the outcome on assets it has never
seen, the domain of validity is defined by market structure and the equity failure becomes a
boundary condition rather than a mystery. If it does not, the edge is a bet on one asset class and
should be described and sized as one.

That is a genuine out-of-sample test of a *theory*, not of a parameter — and it is the first
experiment in this project that could fail in a way that teaches something new.
