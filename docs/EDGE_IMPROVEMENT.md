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
