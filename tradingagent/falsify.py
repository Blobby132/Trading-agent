"""Falsification: what does this machinery produce when there is nothing to find?

Every result in this repository has been compared against buy-and-hold or
against another strategy. None has been compared against **noise**, and that
gap makes several other numbers uninterpretable. A deflated Sharpe of 0.13 is a
probability computed from a formula; whether the formula is calibrated for
*this* pipeline - this search, this selection rule, this walk-forward, this cost
model - is an empirical question nobody has asked.

So the experiments here are built to fail. Each one removes a specific source of
edge and runs the **entire** pipeline on what is left. If the pipeline still
produces the result, the result was never evidence of that edge.

Four nulls, each isolating a different claim
--------------------------------------------

``shuffled_signal``
    Keep the strategy's exposure profile exactly - same weights, same turnover,
    same time in market - and destroy only *when* they occur, by permuting the
    weight series in blocks. Isolates the claim "the timing is informative".
    Blocks rather than single bars because a bar-by-bar shuffle also destroys
    the position's persistence, which would change turnover and costs and make
    the comparison unfair.

``sign_flipped``
    Keep magnitude and timing, randomise direction in blocks. Isolates "the
    strategy knows which way".

``bootstrapped_prices``
    Resample the price path itself with a circular block bootstrap, preserving
    volatility clustering and fat tails while breaking longer-horizon trend
    structure, then run the full search and walk-forward on the synthetic path.
    This is the important one: it measures what the machine produces from data
    whose exploitable structure has been removed, *including* the contribution
    of searching hundreds of configurations over it.

``random_selection``
    Leave the data alone and cripple only the optimiser: on each fold, choose
    the top-k at random instead of by training score. Isolates "the selection
    step adds value" - the one null that tests the search rather than the
    signal.

Reading the output
------------------
The question is never "did the null make money". Over a rising asset a null
that holds exposure will make money, and that is the point: the comparison is
between the **real result and the null distribution**, not between the real
result and zero. A real Sharpe inside the null's central mass is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data import OHLCV_COLUMNS
from .engine import ExecutionConfig
from .optimize import WalkForwardConfig, walk_forward

# --------------------------------------------------------------------------- #
# block resampling primitives
# --------------------------------------------------------------------------- #
def block_permute(values: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Cut the series into contiguous blocks and reorder the blocks.

    Preserves within-block structure (a position that was held for days is still
    held for days) and destroys between-block alignment with anything else.
    """
    n = len(values)
    block = max(1, int(block))
    starts = list(range(0, n, block))
    order = rng.permutation(len(starts))
    pieces = [values[s : min(s + block, n)] for s in starts]
    out = np.concatenate([pieces[i] for i in order])
    return out[:n]


def circular_block_bootstrap(values: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Resample with replacement in blocks, wrapping at the end.

    Circular rather than plain so every observation is equally likely to appear,
    including the ones near the edges - a plain block bootstrap under-samples
    the tails of the series, which for a price history means under-sampling
    whichever era happens to sit at the end.
    """
    n = len(values)
    block = max(1, int(block))
    out = np.empty(n, dtype=values.dtype)
    filled = 0
    while filled < n:
        start = int(rng.integers(n))
        take = min(block, n - filled)
        idx = (start + np.arange(take)) % n
        out[filled : filled + take] = values[idx]
        filled += take
    return out


def bootstrap_ohlcv(
    df: pd.DataFrame, *, block: int = 21, rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """A synthetic OHLCV path with the same short-horizon character, no trend.

    Close-to-close log returns are block-bootstrapped and re-integrated into a
    new price path. Each bar's open/high/low are then rebuilt at their original
    *ratios* to that bar's close, so intrabar shape - the thing ATR and stop
    logic read - survives intact while the long-horizon path does not.

    Volatility clustering and fat tails survive because they live inside a
    block. Multi-month trend does not, because block order is randomised. That
    is exactly the structure a trend model claims to exploit.
    """
    rng = rng or np.random.default_rng(0)
    close = df["close"].to_numpy(dtype=float)
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
    log_ret[0] = 0.0
    shuffled = circular_block_bootstrap(log_ret[1:], block, rng)
    path = close[0] * np.exp(np.concatenate([[0.0], np.cumsum(shuffled)]))

    ratios = {c: (df[c].to_numpy(dtype=float) / close) for c in ("open", "high", "low")}
    out = pd.DataFrame(index=df.index)
    out["close"] = path
    for c in ("open", "high", "low"):
        out[c] = path * ratios[c]
    out["volume"] = df["volume"].to_numpy(dtype=float)
    # re-assert the OHLC bracket; float noise in the ratios can break it
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)
    return out[OHLCV_COLUMNS]


# --------------------------------------------------------------------------- #
# weight-level nulls
# --------------------------------------------------------------------------- #
def shuffle_weights(block: int, seed: int) -> Callable:
    """A ``weight_transform`` that block-permutes the sized weight series.

    Same exposures, same holding periods, same turnover profile - different
    days. Anything the strategy earns that survives this was not earned by
    timing.
    """
    def _transform(w):
        rng = np.random.default_rng(seed)
        if isinstance(w, pd.DataFrame):
            return pd.DataFrame(
                {c: block_permute(w[c].to_numpy(), block, rng) for c in w.columns},
                index=w.index,
            )
        return pd.Series(block_permute(w.to_numpy(), block, rng), index=w.index, name=w.name)

    return _transform


def flip_signs(block: int, seed: int) -> Callable:
    """A ``weight_transform`` that randomises direction in blocks, keeping size."""
    def _transform(w):
        rng = np.random.default_rng(seed)
        n = len(w)
        blocks = int(np.ceil(n / max(1, block)))
        signs = np.repeat(rng.choice([-1.0, 1.0], size=blocks), max(1, block))[:n]
        if isinstance(w, pd.DataFrame):
            return w.mul(pd.Series(signs, index=w.index), axis=0)
        return w * pd.Series(signs, index=w.index)

    return _transform


# --------------------------------------------------------------------------- #
# the suite
# --------------------------------------------------------------------------- #
@dataclass
class NullResult:
    """One replication of one null."""

    null: str
    replication: int
    seed: int
    stats: Dict[str, float]

    def row(self) -> Dict[str, object]:
        s = self.stats
        return {
            "null": self.null,
            "replication": self.replication,
            "seed": self.seed,
            "final_equity": s.get("final_equity"),
            "cagr": s.get("cagr"),
            "sharpe": s.get("sharpe"),
            "max_drawdown": s.get("max_drawdown"),
            "n_trades": s.get("n_trades"),
            "total_return": s.get("total_return"),
        }


def _run_once(
    data: pd.DataFrame,
    base_exec: ExecutionConfig,
    wf: WalkForwardConfig,
    space: Optional[Dict[str, Sequence]],
    *,
    weight_transform=None,
    selection: str = "best",
) -> Dict[str, float]:
    result = walk_forward(
        data, base_exec, wf, space,
        weight_transform=weight_transform, selection=selection,
    )
    stats = result.stats()
    stats["n_evaluations"] = result.n_evaluations
    return stats


def null_suite(
    data: pd.DataFrame,
    *,
    base_exec: Optional[ExecutionConfig] = None,
    wf: Optional[WalkForwardConfig] = None,
    space: Optional[Dict[str, Sequence]] = None,
    nulls: Sequence[str] = ("shuffled_signal", "sign_flipped", "bootstrapped_prices", "random_selection"),
    replications: int = 20,
    block: int = 21,
    seed0: int = 1000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run each null ``replications`` times through the full pipeline.

    Every replication re-runs the search, the fold loop and the cost model, so
    the distribution returned is the distribution of *this pipeline's output*
    under that null - not the distribution of a strategy's returns. That is the
    thing the real result has to be compared against.

    ``block`` is in bars and should exceed the strategy's typical holding period
    for the weight-level nulls, or the permutation will chop positions in half
    and change turnover.
    """
    base_exec = base_exec or ExecutionConfig()
    wf = wf or WalkForwardConfig(verbose=False)
    wf = replace(wf, verbose=False)
    rows: List[NullResult] = []

    for null in nulls:
        for rep in range(replications):
            seed = seed0 + rep
            try:
                if null == "shuffled_signal":
                    stats = _run_once(data, base_exec, wf, space,
                                      weight_transform=shuffle_weights(block, seed))
                elif null == "sign_flipped":
                    stats = _run_once(data, base_exec, wf, space,
                                      weight_transform=flip_signs(block, seed))
                elif null == "bootstrapped_prices":
                    fake = bootstrap_ohlcv(data, block=block, rng=np.random.default_rng(seed))
                    stats = _run_once(fake, base_exec, wf, space)
                elif null == "random_selection":
                    stats = _run_once(data, base_exec, replace(wf, seed=seed), space,
                                      selection="random")
                else:
                    raise ValueError(f"unknown null {null!r}")
            except Exception as exc:                      # noqa: BLE001 - a failure is a result
                if verbose:
                    print(f"[null] {null} rep {rep}: FAILED {type(exc).__name__}: {exc}", flush=True)
                continue
            rows.append(NullResult(null, rep, seed, stats))
            if verbose:
                print(f"[null] {null} rep {rep}: ${stats.get('final_equity', float('nan')):,.0f} "
                      f"Sharpe {stats.get('sharpe', float('nan')):.2f}", flush=True)

    return pd.DataFrame([r.row() for r in rows])


def compare_to_null(
    observed: Dict[str, float], nulls: pd.DataFrame, *, metric: str = "sharpe"
) -> pd.DataFrame:
    """Where the real result sits inside each null distribution.

    ``p_value`` is the one-sided empirical probability that the null produces a
    value at least as large as the observed one, with the standard +1/+1
    correction so a finite number of replications can never report exactly zero.
    With 20 replications the smallest reportable p-value is 1/21 = 0.048, and
    quoting anything smaller would be an artefact of the replication count
    rather than a fact about the strategy.
    """
    value = float(observed.get(metric, float("nan")))
    rows = []
    for null, group in nulls.groupby("null"):
        sample = group[metric].dropna().to_numpy()
        if not len(sample):
            continue
        n_ge = int((sample >= value).sum())
        rows.append({
            "null": null,
            "replications": len(sample),
            f"observed_{metric}": value,
            "null_median": float(np.median(sample)),
            "null_p90": float(np.percentile(sample, 90)),
            "null_max": float(sample.max()),
            "n_null_ge_observed": n_ge,
            "p_value": (n_ge + 1) / (len(sample) + 1),
            "percentile_of_observed": float((sample < value).mean() * 100.0),
        })
    return pd.DataFrame(rows)
