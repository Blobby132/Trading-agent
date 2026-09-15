# Improving the economic edge

*Can the strategy's defensive trend-following advantage be kept while capturing more of the
upside — without overfitting?*

The previous phase concluded the strategy contains genuine information about trend structure but
that the economically useful edge was too small: **+0.031 Sharpe over buy-and-hold, +0.015 over a
constant 40% position**, bought with 962 trades against 73.

---

## 1. Where the return was actually coming from

Before changing anything, the pipeline was instrumented stage by stage.

### The exposure chain, by regime (BTC daily, 2017-07-29 → 2026-09-12)

| regime | share | raw signal `|w|` | vol-target scale | sized | realised (post-engine) | asset return |
|---|---|---|---|---|---|---|
| bull | 51.6% | **0.447** | 1.59 | 0.699 | **0.699** | +98.4%/yr |
| bear | 30.6% | 0.131 | 1.61 | 0.241 | 0.225 | −44.3%/yr |
| sideways | 11.8% | 0.400 | 1.72 | 0.635 | 0.598 | +76.1%/yr |

**The bull-market shortfall is not a risk-layer problem.** In bull regimes the engine removes
**0.0%** of the sized weight — the ATR stops and kill switch cut nothing, because the walk-forward
had already switched them off in 74% and 58% of folds respectively. The loss is upstream, at the
signal: mean conviction is 0.447 of maximum and **33.5% of bull bars are completely flat**. Across
the whole sample **50% of bars are flat** and the median signal is 0.012.

### What the optimiser itself selects

Across 19 folds, the top-1 pick had ATR stops **off** 74% of the time, the regime filter **off**
58%, **equal** (not adaptive) weighting 63%, `allow_short=0` 100%, and `max_leverage` pinned at its
**maximum** 100%. The optimiser was already ablating the sophistication and asking for more
exposure than the cap allowed.

### Agent versus a constant position — decomposed

The agent's +55.7% final-equity advantage over a constant 40% position splits into:

| source | contribution |
|---|---|
| **level** — simply holding more on average (0.400 → 0.505) | **+48.1%** |
| **timing** — the active decisions, at matched average exposure | **+5.1%** |

The timing premium is ~0.55%/yr, and it costs 962 trades against 73. Against a constant position
at the agent's *own* 0.505 average exposure: final equity +5.1%, Sharpe +0.014, **Sortino −0.149**,
drawdown 5.2pp better.

---

## 2. Frozen baseline (the immutable control)

BTC-USD daily, walk-forward, BASE costs, 64 candidates, median of 3 seeds.

| | agent | buy & hold | const 50% | const 40% | momentum 12m | SMA-200 | SMA 50/200 |
|---|---|---|---|---|---|---|---|
| final equity | **$1,029** | $2,857 | $956 | $661 | $1,446 | $923 | $561 |
| CAGR | 29.1% | 44.4% | 28.1% | 23.0% | 34.0% | 27.6% | 20.8% |
| Sharpe | 0.917 | 0.885 | 0.901 | 0.901 | 0.853 | 0.784 | 0.642 |
| Sortino | 1.106 | 1.218 | 1.252 | 1.252 | 1.005 | 0.796 | 0.607 |
| max drawdown | −49.5% | −83.8% | −56.1% | −46.9% | −53.2% | −64.4% | −66.0% |
| trades | 949 | 1 | 73 | 101 | 44 | 67 | 19 |
| avg exposure | 0.387 | 1.00 | 0.50 | 0.40 | 0.61 | 0.49 | 0.49 |
| seed range | $888 – $1,788 | — | — | — | — | — | — |

**That seed range is the yardstick.** Any candidate whose improvement is smaller than it has not
been demonstrated.

---

## 3. Hypotheses tested

### Refuted *before* any code was written

The brief's §6 proposed that the strategy exits profitable trends too early (H1 ATR stops, H2
regime confirmation, H3 slower de-risking). The data refutes all three:

| horizon after an exit | asset return | unconditional | difference |
|---|---|---|---|
| 5 bars | −1.42% | +0.81% | **−2.24%** |
| 10 bars | −0.59% | +1.66% | **−2.25%** |
| 20 bars | +1.72% | +3.38% | **−1.67%** |
| 60 bars | −1.88% | +10.90% | **−12.78%** |

After the agent exits, the asset does **worse** than unconditional at every horizon. Widening
stops, slowing regime confirmation or delaying de-risking would cost money, so none were built.
*Caveat: 45 exits is a small sample.*

### Conviction is informative — non-monotonically

| conviction bucket | bars | mean next-bar return | t | bar Sharpe |
|---|---|---|---|---|
| flat (0) | 1,665 | +16.3 bps | 1.79 | 0.84 |
| weak (0.01–0.25) | 205 | **−17.3 bps** | −0.80 | −1.07 |
| moderate (0.25–0.5) | 244 | +18.3 bps | 1.06 | 1.30 |
| strong (0.5–0.75) | 495 | +4.8 bps | 0.34 | 0.29 |
| **full (>0.75)** | 723 | **+33.0 bps** | **2.45** | **1.74** |

Only the full-agreement bucket is significant. Spearman IC among invested bars +0.052 (n=1,667,
2 s.e. = 0.049) — marginal.

### Against the 12-month momentum baseline

| | share of bars | asset annualised |
|---|---|---|
| both invested | 39.4% | +90.3% |
| **momentum invested, agent flat** | **21.9%** | **+40.2%** |
| **agent invested, momentum flat** | **10.3%** | **−26.3%** |
| both flat | 28.3% | +28.4% |

The agent's *distinctive* positions are net harmful, and the block it misses is large and rising.
Momentum holds far longer: mean spell 93 bars against the agent's 37 (max 1,015 against 212).

---

## 4. Component ablation

Frozen config, full traded window — **diagnostic, in-sample w.r.t. parameter choice.**

| change | Δ final equity | Δ Sharpe | classification |
|---|---|---|---|
| − volatility targeting | **−$1,260** | −0.130 | **KEEP** — much more valuable than the brief suspected |
| + signal smoothing (1→5) | −$1,078 | −0.186 | **KEEP** current value |
| full 6-model ensemble (vs 3) | −$1,173 | −0.175 | **KEEP** the 3-model subset |
| cap gross at 1.0× | −$967 | −0.050 | **KEEP** the 2.0× cap |
| + ATR stop | −$713 | −0.110 | **KEEP** off |
| − drawdown kill switch | +$50 | +0.004 | **NEUTRAL** — no measurable contribution |
| − trailing stop | $0 | 0.000 | **DEAD** — inert whenever `atr_stop_mult = 0` |
| + adaptive weighting | −$59 | −0.029 | **NEUTRAL** — equal weighting is as good |
| + regime filter | +$213 | +0.061 | looked like **MODIFY** — *rejected out of sample, see §5* |
| target_vol 0.5 (vs 0.8) | −$958 | +0.037 | **MIXED** — the Sharpe-for-return trade the brief rejects |

Two findings worth naming: `trail_stop` is a **dead parameter** in the selected configuration, and
the drawdown kill switch contributes nothing measurable.

---

## 5. Results

### Batch 1 — exposure shaping (median of 3 seeds)

| variant | final | CAGR | Sharpe | Sortino | maxDD | exposure | costs | verdict |
|---|---|---|---|---|---|---|---|---|
| V0 baseline | $1,029 | 29.1% | 0.917 | 1.106 | −49.5% | 0.387 | $110.73 | control |
| V1 shape 1.5 | $1,100 | 30.0% | 0.963 | 1.157 | −48.8% | 0.351 | $108.01 | promising |
| V2 shape 2.0 | $1,115 | 30.2% | 0.984 | 1.197 | −45.2% | 0.336 | $102.50 | promising |
| V3 deadband 0.15 | $945 | 27.9% | 0.933 | 1.101 | −51.9% | 0.386 | $116.14 | **REJECT** |
| V4 deadband 0.30 | $1,006 | 28.8% | 0.902 | 1.089 | −49.9% | 0.385 | $114.73 | **REJECT** |
| V5 shape+deadband | $1,086 | 29.9% | 0.961 | 1.162 | −48.8% | 0.356 | $109.84 | worse than shape alone |

Shaping improves every dimension at once and is **monotone** in the exponent, at *lower* exposure
and lower cost. The deadband is rejected — which the statistics predicted, since the bucket that
motivated it had t = −0.80.

### Batch 2 — regime filter and trend floor (median of 3 seeds)

| variant | final | CAGR | Sharpe | Sortino | maxDD | exposure | trades | verdict |
|---|---|---|---|---|---|---|---|---|
| W0 control | $1,029 | 29.1% | 0.917 | 1.106 | −49.5% | 0.387 | 949 | control |
| W1 regime filter ON | $1,078 | 29.8% | **0.866** | **0.959** | −49.5% | 0.389 | 699 | **REJECT** |
| W2 floor 0.25 | $1,735 | 36.7% | 1.032 | 1.341 | −49.1% | 0.469 | 810 | strong |
| **W3 floor 0.50** | **$2,471** | **42.1%** | **1.089** | **1.454** | −46.7% | 0.551 | 711 | **ACCEPTED** |
| W4 regime + floor | $2,290 | 40.9% | 1.036 | 1.237 | −50.7% | 0.481 | 611 | worse than floor alone |
| W5 regime + floor + shape | $2,162 | 40.0% | 1.064 | 1.324 | −48.1% | 0.462 | 696 | worse than floor alone |

**The regime filter is the instructive rejection.** The in-sample ablation showed it improving
return, Sharpe, drawdown and turnover *simultaneously*. Out of sample it **lowers** Sharpe to 0.866
and Sortino to 0.959. Ablation misled; the walk-forward caught it. This is why ablation stays
diagnostic and acceptance runs on walk-forward.

### The falsification test — is the floor just the momentum baseline?

If performance kept climbing to floor = 1.00 ("always long while 12-month momentum is up"), the
floor would simply *be* the one-line rule and the agent would contribute nothing. Six seeds:

| floor | median final | Sharpe | Sortino | maxDD | exposure | paired vs control |
|---|---|---|---|---|---|---|
| 0.00 | $1,119 | 0.917 | 1.090 | −48.2% | 0.389 | — |
| 0.50 | $2,475 | 1.088 | 1.443 | **−47.0%** | 0.545 | **+113.8%, 6/6, t=+5.26** |
| 0.75 | $2,606 | 1.091 | 1.466 | −50.8% | 0.581 | +128.0%, 6/6, t=+5.20 |
| **1.00** | $2,256 | 1.020 | 1.368 | −51.7% | 0.598 | +86.4%, 6/6, t=+4.33 |

**It peaks at 0.75 and falls at 1.00.** Leaving the agent room to size around the floor beats
forcing full exposure, so the signal is contributing on top of the floor. The test could have
ended this line of work and did not.

**0.50 was adopted, not the 0.75 peak.** The two are statistically indistinguishable; 0.50 has the
better drawdown and sits further from the degenerate end. Taking the maximum of a grid is the
selection bias this exercise exists to avoid.

---

## 6. Before vs after

Walk-forward OOS, BTC-USD daily, BASE costs, seed 0.

| | baseline | **+ trend floor 0.50** | change |
|---|---|---|---|
| final equity | $1,029 | **$2,479** | **+141%** |
| CAGR | 29.1% | **42.1%** | +13.0 pp |
| Sharpe | 0.917 | **1.089** | +0.172 |
| Sortino | 1.106 | **1.470** | +0.364 |
| max drawdown | −51.3% | **−45.9%** | **4.4 pp better** |
| trades | 962 | **711** | 26% fewer |
| avg exposure | 0.387 | 0.551 | +42% |
| **vs matched constant** | +5.1% eq, +0.014 Sh, **−0.149 So** | **+106.3% eq, +0.171 Sh, +0.188 So** | the test the baseline failed |
| vs constant 40% | +55.7% | **+275.2%** | |
| vs momentum 12m | −28.9% | **+71.4%** | now beats it |
| vs buy & hold | −64.0% | −13.2% | still behind on return, +0.203 Sharpe |
| bull capture | 8.29 / 24.26 = **34%** | 19.68 / 24.26 = **81%** | upside recovered |
| bear excess | +0.418 | **+0.459** | defence preserved |
| drift-controlled edge vs B&H | **+0.031** | **+0.203** | **6.5×** |

This is the profile the brief asked for: return *up*, risk-adjusted return *up*, drawdown *lower*,
turnover *lower*. It is not a Sharpe-for-return trade.

---

## 7. Robustness of the accepted change

| test | result |
|---|---|
| **Parameter stability** | median 0.989, **17/18 on a plateau**; only `allow_short` spikes (a binary flag whose other setting is simply worse) |
| **Cost adversity** | beats the matched constant to **~100 bps/side** (was ~75) |
| **Null: shuffled signal** | p = 0.048, observed above all 20 replications |
| **Null: drift-controlled bootstrap** | real edge **+0.203** vs null median −0.330; 0/20 null ≥ observed, p = 0.048 |
| **Regimes** | still positive excess in **1 of 3** — bear only |
| **Cross-asset** | **still 0/4** unseen assets beat buy-and-hold |
| **Look-ahead** | future-poisoning tests on every new code path |
| **Execution parity** | unchanged, passing |

### Cross-asset, before and after — reported, not hidden

| asset | tier | before | after | buy & hold | beats B&H |
|---|---|---|---|---|---|
| BTC-USD | trained | $53,215 | $126,200 | $27,554 | ✓ |
| ETH-USD | validation | $22,770 | $22,497 | $18,028 | ✓ |
| SPY | unseen | $201 | $186 | $351 | ✗ |
| QQQ | unseen | $412 | $581 | $612 | ✗ |
| DIA | unseen | $95 | $128 | $286 | ✗ |
| IWM | unseen | $93 | $83 | $238 | ✗ |

**The improvement is crypto-specific.** It roughly doubles the trained asset, leaves the validation
asset unchanged, and is mixed on unseen equities. A plausible mechanism: the floor keys off the
12-month trend, which on US equities over 2016-2026 is up almost continuously, so the floor
degenerates into near-constant levered exposure — the very configuration the previous phase showed
fails there.

---

## 8. The acceptance gate

| check | baseline | modified |
|---|---|---|
| oos_performance | PASS | PASS |
| **beats_matched_baseline** | **FAIL** | **PASS** |
| parameter_stability | PASS | PASS |
| cost_robustness | PASS | PASS |
| regime_robustness | FAIL | FAIL |
| cross_asset | FAIL | FAIL |
| statistical_evidence | FAIL | FAIL |
| capacity | PASS | PASS |
| execution_realism | PASS | PASS |
| **blocking failures** | **4** | **3** |

### The honest cost of the search

| | baseline | modified |
|---|---|---|
| deflated Sharpe at 1,216 trials (like-for-like) | 0.288 | **0.484** |
| deflated Sharpe at 19,456 trials (what this phase actually searched) | — | **0.228** |

At matched trial counts the deflated Sharpe nearly doubles. Counting the 16 variants actually
tried, it *falls below* the baseline's. **The strategy got better and the evidence for it got
weaker at the same time**, because finding the improvement cost statistical credibility. Both
numbers are reported because quoting only the first would be the oldest trick in this field.

---

## 9. Remaining weaknesses

1. **Does not generalise.** 0/4 unseen equity assets, unchanged.
2. **One regime.** Still adds value only in bear markets relative to benchmark.
3. **Deflated Sharpe below a coin flip** once this phase's search is counted.
4. **The floor is an externally-motivated import.** 12-month time-series momentum is a documented
   anomaly with decades of cross-market evidence (Moskowitz-Ooi-Pedersen) — that external prior is
   what makes it defensible rather than curve-fit — but it is still an ingredient chosen *after*
   looking at where this agent failed against that exact baseline.
5. **Exposure rose 42%.** The matched-exposure control says the gain is not merely more risk, but
   the account is meaningfully more exposed and its drawdown remains ~−46%.
6. **One asset, one era, two crypto bull markets.** Unchanged and unfixable from here.

---

## 10. Conclusion — did we improve the edge, or just the backtest?

**The edge, on BTC — and the backtest as well.** The distinction is carried by four results that a
mere backtest improvement would not have produced:

1. **The matched-exposure control passes.** +106% equity, +0.171 Sharpe, +0.188 Sortino against a
   constant position at the *same average exposure*. The baseline scored +5.1%, +0.014 and −0.149.
   This is the test that separates timing skill from simply holding more.
2. **The drift-controlled null.** The measured edge over buy-and-hold on trend-free synthetic paths
   rose from +0.031 to +0.203, with 0 of 20 null replications reaching it.
3. **The falsification test passed.** Performance peaks at floor 0.75 and *falls* at 1.00, so the
   agent is contributing on top of the momentum floor rather than being replaced by it.
4. **Paired seed testing.** 6/6 seeds, t = +5.26.

**What did not improve: generalisation.** Still 0/4 unseen assets, still one regime, and the
deflated Sharpe falls once this phase's own search is honestly counted.

**Recommendation.** Adopt `trend_floor = 0.50` for the BTC daily configuration, with the
cross-asset failure recorded alongside it. Do **not** claim it as a general improvement to the
mechanism — the evidence supports a crypto-specific gain and explicitly does not support transfer.
The paper account is unchanged; nothing here is strong enough to retune a running experiment.
