# Paper trading runbook

The account is **open**. This is the operating manual.

## What is being traded

| | |
|---|---|
| Strategy | fixed 12-1 momentum, top decile, long only, never adapts |
| Universe | `us_large_cap` (124 names) |
| Capital | $100 |
| Rebalance | monthly |
| Costs | BASE scenario — 15 bps per side |
| Data | `nasdaq` (Yahoo was rate-limiting; see *Known gaps*) |
| Opened | 2026-09-13, orders decided on the 2026-09-11 close |

This is the simplest strategy in the repository, and the one that beat every
adaptive variant across three rounds of testing. It was chosen because it has a
thirty-year prior behind it and carries no multiple-testing penalty — nothing
about it was fitted to this sample.

## The cadence

Run this **once a month**, after a trading day's close:

```bash
python -m tradingagent.paper rebalance
```

Each run does two things, in this order:

1. **fills** whatever was queued last time, against the current bar's open;
2. **decides** a new basket from the latest close and queues it for next time.

That is the same one-bar lag the backtest models — you see a close, you place an
order, it fills at the next open. A run cannot fill orders it decided in the
same invocation, and `fill_pending` raises if asked to.

Between rebalances, optionally mark the account so the equity curve has more
points (it makes the tracking-error estimate better, and costs nothing):

```bash
python -m tradingagent.paper mark
```

Any time:

```bash
python -m tradingagent.paper positions    # what is held, and at what basis
python -m tradingagent.paper report       # live vs backtest
```

## What to watch, in this order

The report deliberately puts P&L last.

1. **Return correlation** — the paper account should move day-to-day *with* its
   own backtest. Above ~0.9 means the model is being executed as designed.
2. **Tracking error** — if the live account and a backtest of the identical
   strategy over the identical dates disagree, the execution assumptions are
   wrong somewhere. That is a finding about the framework, not about the market.
3. **P&L** — last, because six months of it cannot settle anything. A paper
   account that makes money while behaving nothing like its backtest has told
   you the backtest is wrong, not that the strategy works.

## What the backtest predicts

Measured on the same configuration over 2019-10 → 2026-09, BASE costs:

| | |
|---|---|
| CAGR | 31.4% |
| Sharpe | 1.09, 90% interval **[0.56, 1.68]** |
| Max drawdown | −32%, and the bootstrap's 5th percentile is −49% |
| Turnover | low — roughly 12 positions, held ~130 days on average |

**Read the interval, not the point estimate.** A Sharpe that could plausibly be
0.56 is a different proposition from one that is 1.68, and six months of paper
trading will not distinguish them either. What six months *can* do is show
whether the execution model is honest.

Expect drawdowns. A −32% drawdown is the central expectation, not the bad case.

## Known gaps in this run

- **Nasdaq data is price-return, not total-return.** Dividends are missing, so
  the paper equity understates real total return by roughly the universe's yield
  (~1.5%/yr). The comparison backtest uses the same source, so the *divergence*
  numbers stay apples-to-apples — but the absolute return is understated.
  Switch to `--source yahoo` when it is reachable.
- **The universe is not point-in-time.** Names that delisted are absent, and
  membership is today's list. See the README's Known limitations.
- **The account models no stops or kill switch**, which matches this strategy
  exactly. A stop-using strategy would diverge from its backtest and nothing
  currently detects that.

## If something looks wrong

- **Tracking error above ~10% annualised** with correlation below 0.9: the fills
  are not matching. Check that the rebalance cadence matches the backtest's
  `rebalance_every`, and that both use the same `periods_per_year` (252 for
  equities, 365 for crypto — a mismatch shows up as a financing difference).
- **`fill_pending` raised**: it was asked to fill on the same bar the orders
  were decided on. That is the guard working. Wait for the next bar.
- **No new orders**: the book already matches the target basket. Normal between
  monthly rebalances.

## State

`results/paper_account.json` holds positions, cash, every fill and the full
equity history. It is **tracked in git on purpose** — a live experiment has to
survive a machine being rebuilt. Commit it after each rebalance:

```bash
git add results/paper_account.json && git commit -m "paper: rebalance $(date +%F)"
```

## The one rule

Do not change the strategy because paper trading had a bad month. The whole
point of running it forward is that it is the only test left that nothing in
this repository can fool — and a test you adjust while it runs is not a test.
