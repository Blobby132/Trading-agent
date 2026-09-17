# Independent-market validation of `BTC_FLOOR_050_CONTROL`

**Classification: `PARTIALLY SURVIVES`** — the mechanism travels to the other crypto and to bear
regimes everywhere, and fails everything else. Detail in §14; the reasoning is §13.

Validation only. No parameter was tuned, no configuration was searched, no asset was dropped, and
the production code was not modified. `git diff` on `control.py`, `agent.py`, `risk.py` and
`engine.py` is empty for this phase.

---

## 1. Experiment identity and reproducibility fingerprint

| | |
|---|---|
| experiment ID | `INDEPENDENT_EDGE_VALIDATION` |
| commit SHA | `02768f44001eea95b2108c93699531788020ee33` |
| code fingerprint (sha256 of 6 core modules) | `f7ad0666b59f405a` |
| control | `BTC_FLOOR_050_CONTROL` |
| seed | 0 (the vector's source walk-forward) |
| execution | decision at close of bar *t*, **fill at open of bar t+1** |
| costs | BASE = 15 bps/side (10 fee + 2.5 half-spread + 2.5 slippage) |
| python / pandas | 3.11 / (recorded in `results/validation/fingerprint.json`) |

**Control verified against its pinned metrics.** Regenerated now: final equity **$2,479.37**,
Sharpe **1.0885**, Sortino 1.4702, maxDD −45.9%, 711 trades, traded 2017-07-29 → 2026-09-12. These
match `tradingagent/control.py` exactly — the control is reproducible.

**No adaptation to the test asset.** `apply_frozen` calls `sized_weight_for` with a fixed parameter
dict and runs the engine. There is no search, no fit, and no fold loop in the path, so nothing
about a test asset can reach parameter selection.

### The frozen vector — identical on every asset

```
allocation inverse_vol   allow_short 0        atr_stop_mult 0.0     donchian_entry 40
ema_fast 20              ema_slow 100         max_drawdown_stop 0.35  max_leverage 2.0
max_positions 0          min_trade_frac 0.1   mom_lookback 60       perf_lookback 60
regime_filter 0          regime_trend 200     signal_smooth 1       softmax_temp 4.0
strategies 2             target_vol 0.8       trail_stop 1          weighting equal
trend_floor 0.50         trend_floor_lookback 252
```
Chosen in 7 of 19 folds of the control's own walk-forward.

---

## 2. Data availability

| asset | tier | source | bars | window |
|---|---|---|---|---|
| BTC-USD | **development** | Coinbase | 4,073 | 2015-07-20 → 2026-09-12 |
| ETH-USD | unseen crypto | Coinbase | 3,758 | 2016-06-01 → 2026-09-14 |
| SPY / QQQ / DIA / IWM | unseen equity | Nasdaq | 2,513 each | 2016-09-14 → 2026-09-14 |

**Equity data is price-return, not total-return** (Nasdaq is split- but not dividend-adjusted).
Dividends of roughly 1.3–2%/yr are missing from *both* the strategy and its benchmarks, but
buy-and-hold is invested 100% of the time and the strategy less, so restoring dividends would
**widen** the gaps reported below. The equity failures are understated, not overstated.

---

## 3. Era holdout — there isn't one, and it is not claimed

The BTC dataset ends 2026-09-12 and the paper account opened 2026-09-13. The only genuinely
untouched forward period is a handful of days: statistically worthless.

**No untouched chronological holdout of meaningful length exists.** This document is therefore
labelled `INDEPENDENT MARKET VALIDATION` and not `UNTOUCHED HOLDOUT`. Re-running the same BTC
history does not become independent by being re-run.

---

## 4. Test A — the exact frozen configuration

*What happens if we literally deploy the BTC-developed strategy unchanged?*

| asset | tier | final | CAGR | Sharpe | Sortino | maxDD | **avg exposure** | trades |
|---|---|---|---|---|---|---|---|---|
| BTC-USD | development | $126,200 | 89.7% | 1.412 | 1.827 | −57.3% | 0.786 | 611 |
| ETH-USD | unseen crypto | $22,497 | 69.2% | 1.219 | 1.549 | −63.5% | 0.526 | 570 |
| SPY | unseen equity | $186 | 6.4% | 0.370 | 0.410 | −51.4% | **1.620** | 77 |
| QQQ | unseen equity | $581 | 19.3% | 0.701 | 0.836 | −45.0% | **1.563** | 139 |
| DIA | unseen equity | $128 | 2.5% | 0.226 | 0.254 | −47.9% | **1.615** | 92 |
| IWM | unseen equity | $83 | −1.8% | 0.092 | 0.114 | −58.9% | **1.330** | 208 |

**Read the exposure column.** `target_vol = 0.8` on an equity index realising ~18% demands 4.4×
exposure and is capped at 2.0, so the frozen configuration runs the equities at **1.3–1.6× leverage
permanently**. Test A on equities is not a test of the signal; it is a test of a levered signal.

---

## 5. Test B — gross capped at 1.0× (leverage-confound diagnostic)

Not a new strategy and not tuned. One number changed, for all assets, to make the comparison fair.

| asset | final | CAGR | Sharpe | Sortino | maxDD | avg exposure |
|---|---|---|---|---|---|---|
| BTC-USD | **$20,091** | 60.9% | 1.277 | 1.613 | −53.4% | 0.631 |
| ETH-USD | $21,275 | 68.3% | 1.273 | 1.621 | −52.3% | 0.491 |
| SPY | $201 | 7.3% | 0.571 | 0.624 | −29.3% | 0.846 |
| QQQ | $346 | 13.2% | 0.748 | 0.866 | −32.9% | 0.852 |
| DIA | $159 | 4.7% | 0.416 | 0.466 | −26.2% | 0.844 |
| IWM | $121 | 2.0% | 0.202 | 0.250 | −36.1% | 0.704 |

Two things jump out:

* **BTC's headline collapses: $126,200 → $20,091, a −84% fall.** Much of the control's spectacular
  figure is leverage, not signal.
* **ETH barely moves: $22,497 → $21,275, −5%.** ETH's volatility is BTC-like, so the cap rarely
  binds. ETH's result is *not* a leverage artifact.
* Every equity improves on Sharpe and roughly halves its drawdown under the cap.

---

## 6. Baseline comparison — identical bars, capital, costs and execution

Test A. Bold marks where the strategy wins.

| asset | strategy | B&H | 40% | 50% | mom12 | SMA-200 | SMA 50/200 | matched |
|---|---|---|---|---|---|---|---|---|
| BTC-USD | **$126,200** | $27,554 | $1,870 | $3,512 | $23,614 | $15,428 | $12,335 | $15,058 |
| ETH-USD | **$22,497** | $18,028 | $2,596 | $4,552 | $17,095 | **$78,144** | $22,606 | $5,554 |
| SPY | $186 | $352 | **$174** | $197 | $209 | $198 | $211 | $431 |
| QQQ | $581 | $612 | **$219** | **$268** | **$309** | **$416** | **$450** | $869 |
| DIA | $128 | $286 | $160 | $178 | $137 | $148 | $135 | $311 |
| IWM | $83 | $238 | $151 | $163 | $152 | $143 | $93 | $218 |

| beats… | all 6 | **unseen 5** |
|---|---|---|
| buy-and-hold | 2/6 | **1/5** |
| constant 40% | 4/6 | 3/5 |
| constant 50% | 3/6 | 2/5 |
| 12-month momentum | 3/6 | 2/5 |
| price > SMA-200 | 2/6 | 1/5 |
| SMA 50/200 | 2/6 | 1/5 |
| **exposure-matched** | **2/6** | **1/5** |

Note ETH: a one-line "price above the 200-day average" rule made **$78,144** against the strategy's
$22,497 — 3.5× more. The strategy beats buy-and-hold on ETH and loses badly to the simplest trend
rule there.

---

## 7. Exposure-matched comparison — the test that isolates decision quality

The constant baseline is set **mechanically** to each asset's own realised average exposure under
Test A. Nothing is optimised.

| asset | tier | matched at | strategy | matched | equity edge | **Sharpe edge** | **Sortino edge** |
|---|---|---|---|---|---|---|---|
| BTC-USD | development | 0.786 | $126,200 | $15,058 | **+738%** | **+0.303** | **+0.324** |
| ETH-USD | unseen crypto | 0.526 | $22,497 | $5,554 | **+305%** | **+0.183** | +0.018 |
| SPY | unseen equity | 1.620 | $186 | $431 | **−57%** | **−0.284** | **−0.387** |
| QQQ | unseen equity | 1.563 | $581 | $869 | **−33%** | **−0.094** | **−0.188** |
| DIA | unseen equity | 1.615 | $128 | $311 | **−59%** | **−0.322** | **−0.405** |
| IWM | unseen equity | 1.330 | $83 | $218 | **−62%** | **−0.317** | **−0.434** |

**Both cryptos beat their matched constant on equity and Sharpe. All four equities lose on all
three metrics.** ETH's Sortino edge is +0.018 — effectively zero, so on downside-risk terms even
ETH is only a draw.

---

## 8. Independent time periods

Rule fixed before looking: each asset's usable window split into **three equal chronological
thirds**, 400 bars of warm-up before each. No period chosen for its result.

| beats buy-and-hold | count |
|---|---|
| early third | 2/6 |
| middle third | 2/6 |
| recent third | 2/6 |
| **crypto (BTC + ETH)** | **5/6** |
| **equity (SPY/QQQ/DIA/IWM)** | **1/12** |

**The split is by asset class, not by era.** Every period bucket is 2/6, and the two winners in
each are the two cryptos. This is the single cleanest result in the document: the failure on
equities is not a bad decade, it is the asset class. BTC beat buy-and-hold in 3/3 of its own
independent thirds; ETH in 2/3, missing the recent third by $0.57.

---

## 9. Regime independence

Excess return versus buy-and-hold, by causal trend regime (Test A exposure):

| asset | bear | bull | sideways |
|---|---|---|---|
| BTC-USD | **+0.627** | +633.96 | −0.625 |
| ETH-USD | **+0.542** | −1750.20 | −0.441 |
| SPY | **+0.242** | +3.502 | −1.184 |
| QQQ | **+0.020** | +8.011 | −0.543 |
| DIA | **+0.203** | +1.869 | −0.990 |
| IWM | **+0.098** | +0.998 | −1.009 |

**Bear-market excess is positive on all six assets. Sideways excess is negative on all six.** That
consistency is real evidence of a travelling mechanism: the defensive behaviour is not
BTC-specific.

The bull column must be read with care — at 1.3–1.6× leverage a strategy beats an unlevered
benchmark in a rising market almost mechanically, so the positive equity bull figures are largely
the leverage confound, not skill.

Volatility regimes: high-volatility excess is **negative on all six assets** (−0.26 to −4.37).
The strategy is consistently worse than buy-and-hold when volatility is high, everywhere.

---

## 10. Cost robustness

| asset | LOW | BASE | HIGH | retains | beats B&H at LOW → HIGH |
|---|---|---|---|---|---|
| BTC-USD | $159,733 | $126,200 | $76,507 | 48% | ✓ → **✓** |
| ETH-USD | $29,502 | $22,497 | $15,949 | 54% | ✓ → **✗** |
| SPY | $240 | $186 | $133 | 56% | ✗ → ✗ |
| QQQ | $763 | $581 | $391 | 51% | ✓ → **✗** |
| DIA | $173 | $128 | $66 | 38% | ✗ → ✗ |
| IWM | $121 | $83 | $52 | 43% | ✗ → ✗ |

Sharpe under HIGH costs goes **negative** on DIA (−0.037) and IWM (−0.064).

**Under HIGH costs, only BTC still beats buy-and-hold.** ETH's advantage over buy-and-hold does not
survive adverse execution, though its advantage over the *matched constant* is a separate and
stronger claim.

---

## 11. Parameter invariance (±20%, diagnostic only — nothing adopted)

Final equity relative to the frozen value:

**`trend_floor` (0.40 / 0.50 / 0.60)** — broadly stable. Worst neighbour ≥ 0.73 everywhere;
BTC 0.727 and QQQ 0.766 are moderately sensitive, the rest ≥ 0.82.

**`trend_floor_lookback` (202 / 252 / 302)** — weaker:

| asset | worst neighbour | verdict |
|---|---|---|
| BTC-USD | 0.585 | moderately sensitive |
| ETH-USD | 0.602 | moderately sensitive |
| SPY | 0.849 | broadly stable |
| QQQ | 1.031 | broadly stable |
| DIA | 0.854 | broadly stable |
| IWM | **0.480** | **extremely sensitive** |

The lookback sensitivity flagged in the previous phase **reproduces on ETH** (0.602), an asset that
had no part in choosing it. That is mild evidence the sensitivity is a property of the mechanism
rather than of BTC noise — but it remains the configuration's weakest parameter, and the equities
are insensitive to it mainly because the strategy does not work there at all.

---

## 12. Null / placebo diagnostic

20 replications each, frozen configuration, existing `falsify` framework. p = 0.0476 is the floor
at 20 replications; nothing smaller is claimable.

| asset | null | observed | null median | null max | p |
|---|---|---|---|---|---|
| BTC-USD | shuffled signal | 1.412 | 0.752 | 1.035 | **0.048 ✓** |
| BTC-USD | sign-flipped | 1.412 | −0.114 | 0.614 | **0.048 ✓** |
| BTC-USD | bootstrap, drift-controlled | **+0.321** | −0.129 | +0.185 | **0.048 ✓** |
| ETH-USD | shuffled signal | 1.219 | 0.689 | 1.088 | **0.048 ✓** |
| ETH-USD | sign-flipped | 1.219 | 0.068 | 0.876 | **0.048 ✓** |
| ETH-USD | bootstrap, drift-controlled | **+0.212** | −0.323 | +0.131 | **0.048 ✓** |
| SPY | shuffled signal | 0.370 | **0.493** | 0.814 | **0.857 ✗** |
| SPY | sign-flipped | 0.370 | −0.760 | −0.187 | 0.048 ✓ |
| SPY | bootstrap, drift-controlled | **−0.426** | −0.257 | +0.100 | **0.810 ✗** |

**On SPY, randomly permuting the strategy's own weights produces a *better* result than the
strategy's actual timing** (0.493 against 0.370). Its timing on equities is worse than noise. It
still knows *direction* there (sign-flip passes), which is why it makes money at all — but the
decision of *when* is actively harmful.

Both cryptos clear all three nulls, including the drift-controlled comparison that cancels path
difficulty.

---

## 13. The four hypotheses, classified by the evidence

**Hypothesis A — the trend floor captures a general cross-market phenomenon.** *Partially
supported.* Bear-regime excess is positive on all six assets and sideways excess negative on all
six: the defensive component genuinely travels. But it does not produce a net advantage outside
crypto.

**Hypothesis B — the improvement is primarily crypto-specific.** *Strongly supported.* 5/6 crypto
asset-periods beat buy-and-hold against 1/12 equity ones, with no era pattern. Both cryptos clear
all three nulls; SPY fails two. Both cryptos beat their matched constant; all four equities lose to
theirs.

**Hypothesis C — the improvement is a volatility/leverage/exposure artifact.** *Supported for BTC,
refuted for ETH.* Capping gross at 1.0× cuts BTC from $126,200 to $20,091 (−84%) and ETH from
$22,497 to $21,275 (−5%). BTC's headline is substantially leverage; ETH's is not. This is the
sharpest discriminator in the study, and it says the *mechanism* survives the cap even though BTC's
*magnitude* does not.

**Hypothesis D — largely explained by simple momentum / partial exposure.** *Refuted for BTC,
partially supported for ETH.* Against the exposure-matched constant, BTC is +738% equity and +0.303
Sharpe and ETH +305% and +0.183 — so it is not merely partial exposure. But on ETH a one-line
SMA-200 rule earned $78,144 against the strategy's $22,497, so on that asset a simple rule is
plainly better even though the strategy is not *reducible* to it.

---

## 14. Final classification

### `PARTIALLY SURVIVES`

The mechanism appears in some independent markets and regimes, but not broadly enough to call it
general.

**For it:** it transfers to ETH — an asset that took no part in selecting it — beating ETH's
matched constant by +305% equity and +0.183 Sharpe and clearing all three nulls at the 20-rep
floor. ETH's result is nearly untouched by the leverage cap. And the defensive component
(positive bear excess, negative sideways excess) is consistent across all six assets, which is the
signature of a real behavioural mechanism rather than a fit.

**Against it:** it fails on all four equity indices on every metric that matters — worse than the
matched constant on equity, Sharpe and Sortino in each case; 1/12 independent equity periods beat
buy-and-hold; on SPY its timing is beaten by a random permutation of its own weights. Under HIGH
costs only BTC still clears buy-and-hold. And BTC's headline number is 84% leverage.

It is not `SURVIVES` (four of five unseen assets fail), not `BTC-SPECIFIC` (ETH is genuinely unseen
and genuinely works), not `EXPOSURE/LEVERAGE-DRIVEN` (ETH survives the cap; BTC's *magnitude*
doesn't, but its matched-exposure edge does), and not `NOT DISTINGUISHABLE FROM SIMPLE BASELINES`
(it beats matched exposure decisively on both cryptos and clears the nulls there).

**The honest one-line summary: this is a crypto trend-following edge, not a general one.**

---

## 15. Statistical credibility

| quantity | value |
|---|---|
| accumulated selection burden (phases 1+2) | 23 variants × 1,216 = **27,968 configurations** |
| BTC control deflated Sharpe at that burden | **0.2037** |
| ETH frozen, Test A, raw Sharpe | 1.219 |
| ETH frozen, Test B, raw Sharpe | 1.273 |

**The trial count is not reset for this phase.** No configuration was selected here, so the
selection burden is unchanged at 27,968 — but this phase did *evaluate* 24 further parameter
perturbations and 12 asset-level runs, and none of them is claimed as a discovery.

**ETH carries no selection burden of its own.** The configuration was chosen on BTC and applied
unchanged, so the appropriate statistic for ETH is its null p-value (0.048, the replication floor)
rather than a deflated Sharpe — a deflated Sharpe would be correcting ETH for a search that never
touched it. That is why ETH is the most informative number in this document despite BTC's larger
figures.

A coin scores 0.50. The BTC control sits at 0.20 and this phase did not move it.

### Tests evaluated in this phase

| | count |
|---|---|
| assets | 6 |
| frozen-configuration backtests (Tests A + B) | 12 |
| baseline runs | 42 (7 per asset) |
| exposure-matched runs | 6 |
| independent time-period runs | 18 |
| cost-scenario runs | 18 |
| parameter-perturbation runs | 24 |
| null replications | 180 (3 assets × 3 nulls × 20) |
| **new configurations selected** | **0** |

---

## 16. Limitations

1. **No untouched chronological holdout exists** (§3). Every result is same-history.
2. **Equity data is price-return**, so equity buy-and-hold is understated and the equity failures
   are worse than shown.
3. **One crypto validation asset.** ETH is highly correlated with BTC; it is a genuine out-of-sample
   *asset* but a weak out-of-sample *regime*. Two cryptos is not "multiple independent markets".
4. **Equity window 2016–2026 is one long bull market**, the regime a defensive trend follower will
   lag by construction. That mitigates the underperformance without excusing the outright losses on
   DIA and IWM.
5. **20 null replications** caps every p-value at 0.048.
6. **`trend_floor_lookback` remains the weakest parameter**, moderately sensitive on both cryptos.

---

## 17. What this means, and the next research question

The next question is **not** another parameter, another asset sweep, or another BTC hypothesis. The
evidence now points somewhere specific:

> **Is the edge a property of *crypto*, or a property of *high-volatility, high-trend-persistence
> assets* generally?**

This is answerable and it is the only question whose answer would change what we believe. ETH
working and SPY failing is consistent with two very different stories: "crypto is special" (a
market-structure claim about a young, retail-dominated, 24/7 asset class), or "this needs assets
whose volatility and trend persistence resemble BTC's" (a statistical claim that would predict
success on commodities, EM currencies, small-cap growth or volatility products, and failure on
large-cap indices regardless of asset class).

The two stories make **opposite, testable predictions** on the same instruments. Distinguishing
them requires adding high-volatility *non-crypto* assets with the frozen configuration and the 1.0×
cap — and nothing else. If the trend-persistence story holds, the strategy has a definable domain
of validity and the equity failure stops being a mystery and becomes a boundary condition. If the
crypto story holds, the edge is a market-structure bet on one asset class and should be sized and
described as one.

Either answer is worth more than any further optimisation of this configuration.
