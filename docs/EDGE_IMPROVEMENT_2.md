# Edge improvement, phase 2 — searching for a second source of return

**Outcome: nothing was accepted. Five hypotheses were tested and all five were rejected. And the
phase found a weakness in the improvement that *was* accepted last time.**

The brief said "if nothing survives, that is a successful research outcome." This is that outcome,
plus one finding that runs against the previous phase's conclusion.

---

## 1. The frozen control

`BTC_FLOOR_050_CONTROL`, frozen in `tradingagent/control.py` as code rather than a JSON blob, with
tests pinning its name, its single override and its measured metrics. A later phase that wants a
different baseline must name a new control; it cannot drift this one.

| | |
|---|---|
| configuration | repository defaults + `trend_floor = 0.50` (the only deviation) |
| symbol / interval / source | BTC-USD, daily, Coinbase |
| data / traded window | 2015-01-01 loaded; traded 2017-07-29 → 2026-09-12 (3,333 bars) |
| costs | BASE — 15 bps/side (10 fee + 2.5 half-spread + 2.5 slippage) |
| execution | decision at close of bar *t*, fill at **open of bar t+1** |
| search space | 2,304 configurations, 64 candidates/fold, 19 folds |
| seeds | 0–5 |
| safety | long-only by selection (`allow_short=0` in 100% of folds); gross capped by `max_leverage`; **no live trading; paper interlocks untouched** |

| metric | value |
|---|---|
| final equity (median / min / max of 6 seeds) | **$2,475** / $1,995 / $3,141 |
| CAGR | 42.1% |
| Sharpe | 1.088 |
| Sortino | 1.443 |
| max drawdown | −47.0% |
| average exposure | 0.545 |
| trades (seed 0) | 711 |

---

## 2. Where the control makes and loses money

The first attribution pass produced `+7078% missed`, `−100% damage` and a capture ratio of
**−1,865,178**. All three were artifacts of compounding a *selected subset* of bars — compounding
only the up days answers "what if only the good days happened". Redone additively in log space,
where contributions sum correctly across disjoint subsets:

| segment | bars | share | log contribution | % of total |
|---|---|---|---|---|
| invested | 2,889 | 86.7% | +3.252 | +100.3% |
| flat (w < 0.05) | 443 | 13.3% | −0.010 | −0.3% |
| invested, asset rose | 1,487 | 44.6% | +21.754 | — |
| invested, asset fell | 1,402 | 42.1% | −18.502 | — |

**The floor already closed the gap the last phase opened.** Flat bars contribute −0.3% of total
return and only **1.8%** of them sit inside a 12-month uptrend. Holding every flat bar fully would
have **lost** 0.97 log. There is no money left in "flat while the asset rose" — §4A of the brief is
already solved.

**The floor's bill.** Of bars where the control held >0.5 exposure into a falling market, **87.7%**
had an intact 12-month uptrend. That is the trade the floor makes, and it is the source of the
remaining −45.9% drawdown.

### Where the headroom actually is

Sorted causally by trailing trend strength (90-bar return, rolling 365-bar percentile rank):

| trend strength | bars | mean exposure | asset log | strategy log | capture |
|---|---|---|---|---|---|
| weak | 794 | 0.394 | +0.022 | **−0.054** | — |
| moderate | 829 | 0.488 | −1.155 | −0.060 | good defence |
| strong | 732 | 0.627 | +0.200 | +0.650 | 3.25× |
| **extreme** | 768 | **0.741** | **+2.961** | **+2.193** | **0.74×** |

The control under-captures the strongest trends and **loses money in the weakest**. That asymmetry
— not more leverage — was the target of this phase.

---

## 3. Hypotheses, and what happened to each

Eight candidate conditioning variables were screened causally against the next bar before any code
was written, at a 2 s.e. bar, with the multiple-testing arithmetic stated up front (~0.4 false
positives expected from eight tests).

| feature | IC | q4−q1 | t | cleared? |
|---|---|---|---|---|
| `dist200_rank` | +0.046 | +48.5 bps | 2.53 | ✓ |
| `accel_rank` | −0.039 | **−41.5 bps** | −2.49 | ✓ |
| `trend_rank` | +0.043 | +46.6 bps | 2.46 | ✓ |
| `persist_rank` | +0.009 | +34.9 bps | 2.23 | ✗ (IC not significant) |
| `agree` (multi-horizon) | +0.033 | +33.4 bps | 1.85 | ✗ |
| `vol_rank` | +0.023 | +22.9 bps | 1.29 | ✗ |
| `vol_chg_rank` | +0.005 | −9.2 bps | −0.57 | ✗ |
| `dd_state` | −0.000 | −0.5 bps | −0.03 | ✗ |

`trend_rank` correlates **0.913** with `dist200_rank` — one idea, not two. So two hypotheses
survived screening, and the acceleration one had the **opposite sign to the obvious guess**. A
double sort showed it was not redundant:

| trend quartile | low accel | high accel | spread |
|---|---|---|---|
| weak (t1) | −54.8 bps | −4.6 bps | −50.2 |
| **strong (t4)** | **+57.2 bps** | **−15.6 bps** | **+72.9** |

Steady trends persist; parabolic ones mean-revert. Economically sensible, statistically screened —
and, as it turned out, useless.

### H4 — volatility regime: rejected on evidence, not tested at length

`vol_rank` predicts forward **volatility** (IC +0.256) and not forward **return** (IC +0.023).
Those are different hypotheses and the brief asked for them to be distinguished. Volatility belongs
in the sizing layer, which already uses it. No experiment was spent on it.

---

## 4. Experiments performed

Implemented as an exposure **tilt**: `multiplier = 1 ± k(2r − 1)` on ranks that are uniform by
construction, so the expected multiplier is **1**. It redistributes exposure across states and
cannot raise the cap. Applied *before* the trend floor, so the protected component keeps its
guarantee. Verified exposure-neutral in tests (mean exposure moved 0.3366 → 0.3405).

### Batch 1 — the two screened tilts (4 seeds, paired against the control)

| variant | median final | paired equity | wins | t | Δ Sharpe | exposure |
|---|---|---|---|---|---|---|
| CONTROL | $2,475 | — | — | — | — | 0.553 |
| E1 trend tilt 0.25 | $2,388 | −0.23% | 1/4 | −0.05 | −0.016 | +1.5% |
| E2 trend tilt 0.50 | $2,330 | +5.75% | 2/4 | +0.38 | −0.003 | +0.5% |
| E3 accel tilt 0.25 | $1,877 | **−19.2%** | **0/4** | **−5.01** | −0.056 | −1.3% |
| E4 accel tilt 0.50 | $1,923 | **−19.7%** | **0/4** | **−6.51** | −0.051 | −4.2% |

**Both rejected.** The trend tilt does nothing. The acceleration tilt destroys a fifth of the
return, consistently, in the wrong direction — after clearing a causal screen at t = −2.49.

### Batch 2 — the floor's momentum horizon (4 seeds, paired)

| horizon | median final | paired equity | wins | t | Δ Sharpe |
|---|---|---|---|---|---|
| **12 months (control)** | **$2,475** | — | — | — | — |
| 6 months | $1,735 | −36.3% | 0/4 | −3.52 | −0.120 |
| 9 months | $912 | −49.4% | 0/4 | −3.65 | −0.194 |
| 18 months | $1,351 | −39.9% | 0/4 | −8.55 | −0.099 |

**Rejected** — but see §6, because this result is *not* good news.

---

## 5. Why the tilts failed — the mechanism

Not "the market is efficient". Two measurable reasons:

1. **The tilt had nothing to add.** The control's exposure already correlates **+0.333** with trend
   strength and **−0.152** with acceleration. The agent's component models are trend followers; it
   was already leaning the way both features suggested. Tilting further doubled an existing bet and
   paid turnover for the privilege.
2. **The signal is far too small to survive the conversion.** An IC of −0.0385 explains about
   **0.15%** of next-bar variance. Turning that into a persistent position change costs turnover
   immediately and repays only in expectation over years.

**The general lesson: a feature with a genuine bar-level information coefficient does not
automatically convert into a profitable exposure tilt.** The screen tests whether a variable knows
something; the tilt tests whether that knowledge is *incremental to a book already positioned on
it*, net of the cost of acting.

### The same diagnostic kills §8 (risk layer too conservative)

In the strongest trend quartile the control's exposure spans **0.27** (10th pct) to **1.13** (90th),
with only 21.6% of bars at or above 1.0 and **0.3%** above 1.5. Exposure is **not pinned against a
cap**, so making volatility targeting less conservative would release nothing. The binding
constraint is the signal. No experiment was spent.

---

## 6. ⚠️ A weakness discovered in the *previously accepted* improvement

This is the most important finding of the phase and it argues against the last one.

**`trend_floor_lookback` was never perturbed.** The phase-1 robustness claim — "median stability
0.989, 17 of 18 parameters on a plateau" — covered only parameters in the search space. The floor's
lookback is a config default that never entered the frozen parameter dict, so **the stability test
never touched it.** The 17/18 figure was true and incomplete.

**And the surface around it is not a plateau.** Testing it now:

| horizon | 126 (6m) | 189 (9m) | **252 (12m)** | 378 (18m) |
|---|---|---|---|---|
| median final | $1,735 | **$912** | **$2,475** | $1,351 |

A clean economic optimum would be smooth. This is not: **9-month is worse than both 6-month and
18-month**. The surface is spiky and noisy, and the control sits on the spike.

Two readings, and the honest position is that this evidence cannot separate them:

* **Benign** — 12-month momentum is the horizon the time-series-momentum literature singles out
  across decades and dozens of markets, and 252 was chosen *a priori* from that literature, not
  searched. Finding the a-priori value best is what a real effect looks like.
* **Adverse** — a parameter whose neighbours are 36–49% worse, non-monotonically, is the signature
  of a value fitted to this sample. Its robustness was asserted on a test that never examined it.

**The floor's *level* is robust; the floor's *horizon* is not established.** The level ladder
(0 → 0.25 → 0.50 → 0.75 → 1.00) was smooth, broad and peaked interior. The horizon ladder is a
spike. These are different parameters with different robustness profiles and the previous report
conflated them under one stability figure.

---

## 7. Before vs after

**There is no "after".** No candidate was accepted, so the configuration is unchanged:
`trend_floor = 0.50`, `trend_floor_lookback = 252`, everything else at repository defaults.

| | control | best candidate (E2 trend tilt 0.50) |
|---|---|---|
| median final equity | **$2,475** | $2,330 |
| CAGR | 42.1% | 41.1% |
| Sharpe | 1.081 | 1.064 |
| Sortino | 1.443 | 1.442 |
| max drawdown | −48.3% | −49.2% |
| paired equity vs control | — | +5.75%, **2/4 seeds, t = +0.38** |

The best candidate's paired t-statistic is **+0.38**. Its improvement is far smaller than the
control's own seed range ($1,995–$3,141, a 1.57× spread). Per the brief's own rule — *"a candidate
whose improvement is smaller than the existing seed noise has NOT demonstrated an improvement"* —
it is rejected.

Matched-exposure, cross-asset, regime and cost analyses were **not run** on the candidates. That is
deliberate: nothing cleared the seed-robustness gate, and running the expensive downstream tests on
a candidate that failed the cheap upstream one would be spending budget to dress up a negative
result.

---

## 8. Multiple testing and deflated Sharpe

| trial count | context | deflated Sharpe of the control |
|---|---|---|
| 1,216 | one walk-forward | **0.4845** |
| 19,456 | phase-1 search (16 variants) | **0.2284** |
| **27,968** | **phases 1 + 2 cumulative (23 variants)** | **0.2037** |

**The control's statistical standing got worse during this phase even though the control did not
change.** Seven more configurations were tried against the same data; every one of them was a
chance to find a coincidence, and the correction charges for chances taken, not chances that paid.
Rejecting all seven does not refund it.

A coin scores 0.50. The control is at 0.20.

### Exact experiment count, this phase

| | count |
|---|---|
| conditioning variables screened | 8 |
| distinct configurations walk-forwarded (beyond the control) | 7 |
| walk-forward runs | **37** (20 tilt + 16 horizon + 1 attribution) |
| configurations evaluated inside those runs | ~45,000 |
| hypotheses tested | 5 |
| hypotheses accepted | **0** |
| rejected and recorded | 5 |

Nothing was deleted. Every variant run appears in `results/edge2/`.

---

## 9. Remaining weaknesses

1. **The floor's horizon is unvalidated and sits on a spike** (§6). This is new, and it weakens the
   phase-1 result.
2. **Statistical credibility is now 0.20** and falls with every further experiment on this data.
3. **Still one asset, one era.** Cross-asset generalisation was already 0/4 and nothing here
   addressed it.
4. **Still one regime.** Excess return is still concentrated in bear markets.
5. **The control's drawdown is −47%** and the floor is structurally responsible for much of it —
   87.7% of the damage bars had an intact 12-month uptrend.
6. **Search budget is close to exhausted on this dataset.** The marginal value of another BTC
   hypothesis is now negative: it costs more deflated Sharpe than it can plausibly return.

---

## 10. Final accepted configuration

**Unchanged.** `BTC_FLOOR_050_CONTROL` remains the champion by default, not by merit — nothing beat
it. The paper account is untouched. No safety control was modified.

### What should be tested next

Not another BTC signal. The two questions that would actually move the evidence:

1. **Validate `trend_floor_lookback` out of sample** — on ETH, and on the equity ETFs, using the
   frozen 252. If 12 months is the best horizon *there too*, the a-priori reading of §6 is
   supported and the spike is benign. If the best horizon moves per asset, the floor is fitted and
   phase 1's acceptance should be revisited. **This is the highest-value experiment available and
   it costs one batch.**
2. **Stop searching this dataset.** Every additional hypothesis tested on BTC 2017–2026 lowers the
   deflated Sharpe of whatever survives. The binding constraint on this research is no longer ideas
   — it is that the sample has been looked at too many times.
