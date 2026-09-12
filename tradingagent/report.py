"""Charts and text reports for a finished backtest.

Chart conventions used throughout: one y-axis per plot (never two scales in one
frame), equity on a log scale because the whole point is compounding, a legend
plus a direct end-label whenever more than one series is on screen, and a
recessive grid so the data carries the ink.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# Palette roles. Categorical slots 1-3 are the only ones used together, which
# keeps every pair clear of the colour-vision-deficiency separation floor.
C_AGENT = "#2a78d6"      # slot 1, blue
C_BENCH = "#eb6834"      # slot 2, orange
C_THIRD = "#1baf7a"      # slot 3, aqua
C_GOOD = "#0ca30c"
C_CRITICAL = "#d03b3b"
C_TEXT = "#0b0b0b"
C_TEXT_2 = "#52514e"
C_GRID = "#e3e2de"
C_SURFACE = "#fcfcfb"


def _style(ax):
    ax.set_facecolor(C_SURFACE)
    ax.grid(True, color=C_GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT_2, labelsize=9)
    ax.title.set_color(C_TEXT)
    return ax


def _money(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 100 else f"${x:,.2f}"


def plot_equity(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    *,
    target: float = 1000.0,
    initial: float = 100.0,
    title: str = "Account equity",
    ax=None,
    log: bool = True,
):
    """Equity curve with the goal drawn in, plus the buy-and-hold line."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    _style(ax)

    ax.plot(equity.index, equity.values, color=C_AGENT, linewidth=2.0, label="Agent", zorder=3)
    ax.annotate(
        _money(float(equity.iloc[-1])),
        xy=(equity.index[-1], equity.iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=C_TEXT,
        fontsize=10,
        fontweight="bold",
        va="center",
    )
    if benchmark is not None and len(benchmark):
        b = benchmark.reindex(equity.index).ffill()
        ax.plot(b.index, b.values, color=C_BENCH, linewidth=1.6, label="Buy & hold", zorder=2)
        ax.annotate(
            _money(float(b.iloc[-1])),
            xy=(b.index[-1], b.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=C_TEXT_2,
            fontsize=9,
            va="center",
        )

    ax.axhline(target, color=C_GOOD, linewidth=1.2, linestyle="--", zorder=1)
    # anchored right, clear of the legend in the upper-left corner
    ax.annotate(
        f"target {_money(target)}",
        xy=(equity.index[-1], target),
        xytext=(-4, 5),
        textcoords="offset points",
        color=C_GOOD,
        fontsize=9,
        ha="right",
    )
    ax.axhline(initial, color=C_TEXT_2, linewidth=0.8, alpha=0.5, zorder=1)

    hit = equity[equity >= target]
    if not hit.empty:
        ax.axvline(hit.index[0], color=C_GOOD, linewidth=1.0, alpha=0.6, zorder=1)
        ax.annotate(
            f"target reached\n{hit.index[0].date()}",
            xy=(hit.index[0], float(equity.max())),
            xytext=(6, -22),
            textcoords="offset points",
            color=C_GOOD,
            fontsize=9,
        )

    if log:
        ax.set_yscale("log")
        ax.set_ylabel("equity (USD, log scale)", color=C_TEXT_2, fontsize=9)
    else:
        ax.set_ylabel("equity (USD)", color=C_TEXT_2, fontsize=9)
    ax.set_title(title, fontsize=13, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=C_TEXT_2)
    return ax


def plot_drawdown(equity: pd.Series, *, title: str = "Drawdown from peak", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.6))
    _style(ax)
    dd = (equity / equity.cummax() - 1.0) * 100.0
    ax.fill_between(dd.index, dd.values, 0.0, color=C_CRITICAL, alpha=0.28, linewidth=0)
    ax.plot(dd.index, dd.values, color=C_CRITICAL, linewidth=1.2)
    worst = float(dd.min())
    at = dd.idxmin()
    # label inward when the trough sits near the right edge, or it runs off it
    near_right = dd.index.get_loc(at) > 0.75 * len(dd)
    ax.annotate(
        f"worst {worst:,.1f}%",
        xy=(at, worst),
        xytext=(-6 if near_right else 6, 8),
        textcoords="offset points",
        color=C_CRITICAL,
        fontsize=9,
        fontweight="bold",
        ha="right" if near_right else "left",
    )
    ax.set_ylabel("%", color=C_TEXT_2, fontsize=9)
    ax.set_title(title, fontsize=11, loc="left", pad=8)
    return ax


def plot_exposure(weights: pd.DataFrame | pd.Series, *, title: str = "Gross exposure", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.4))
    _style(ax)
    w = weights.to_frame() if isinstance(weights, pd.Series) else weights
    gross = w.abs().sum(axis=1)
    net = w.sum(axis=1)
    ax.fill_between(gross.index, gross.values, 0.0, color=C_AGENT, alpha=0.20, linewidth=0)
    ax.plot(gross.index, gross.values, color=C_AGENT, linewidth=1.2, label="Gross")
    ax.plot(net.index, net.values, color=C_THIRD, linewidth=1.2, label="Net")
    ax.axhline(0.0, color=C_TEXT_2, linewidth=0.8, alpha=0.5)
    ax.set_ylabel("x equity", color=C_TEXT_2, fontsize=9)
    ax.set_title(title, fontsize=11, loc="left", pad=8)
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=2, labelcolor=C_TEXT_2)
    return ax


def plot_folds(folds: pd.DataFrame, *, title: str = "Out-of-sample return by fold", ax=None):
    """One bar per walk-forward test window - the honest track record."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.2))
    _style(ax)
    rets = folds["return"].to_numpy() * 100.0
    colors = [C_GOOD if r >= 0 else C_CRITICAL for r in rets]
    labels = [str(pd.Timestamp(d).date()) for d in folds["test_start"]]
    ax.bar(range(len(rets)), rets, color=colors, width=0.72)
    ax.axhline(0.0, color=C_TEXT_2, linewidth=0.9)
    ax.set_xticks(range(len(rets)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% return", color=C_TEXT_2, fontsize=9)
    win = float((folds["return"] > 0).mean())
    ax.set_title(f"{title}  -  {win:.0%} of windows positive", fontsize=11, loc="left", pad=8)
    return ax


def plot_monte_carlo(
    returns: pd.Series,
    *,
    initial: float = 100.0,
    target: float = 1000.0,
    n_paths: int = 400,
    block: int = 20,
    seed: int = 0,
    title: str = "Bootstrapped alternative histories",
    ax=None,
):
    """Block-bootstrap fan chart: the same edge, reshuffled into other futures."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.2))
    _style(ax)

    r = returns.dropna().to_numpy()
    r = r[np.isfinite(r)]
    horizon = r.size
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon / block))
    paths = np.empty((n_paths, horizon))
    for p in range(n_paths):
        starts = rng.integers(0, max(r.size - block, 1), size=n_blocks)
        sampled = np.concatenate([r[s : s + block] for s in starts])[:horizon]
        paths[p] = np.maximum(initial * np.cumprod(1.0 + sampled), 0.0)

    x = np.arange(horizon)
    for lo, hi, alpha in ((5, 95, 0.14), (25, 75, 0.22)):
        ax.fill_between(
            x,
            np.percentile(paths, lo, axis=0),
            np.percentile(paths, hi, axis=0),
            color=C_AGENT,
            alpha=alpha,
            linewidth=0,
        )
    ax.plot(x, np.percentile(paths, 50, axis=0), color=C_AGENT, linewidth=2.0, label="Median path")
    ax.plot(
        x,
        initial * np.cumprod(1.0 + r),
        color=C_BENCH,
        linewidth=1.6,
        label="Actual backtest",
    )
    ax.axhline(target, color=C_GOOD, linewidth=1.2, linestyle="--")
    ax.set_yscale("log")
    reached = float((paths.max(axis=1) >= target).mean())
    ax.set_title(
        f"{title}  -  {reached:.0%} of {n_paths} paths touched {_money(target)}",
        fontsize=11,
        loc="left",
        pad=8,
    )
    ax.set_ylabel("equity (USD, log scale)", color=C_TEXT_2, fontsize=9)
    ax.set_xlabel("bars", color=C_TEXT_2, fontsize=9)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=C_TEXT_2)
    return ax


def tearsheet(
    equity: pd.Series,
    *,
    benchmark: Optional[pd.Series] = None,
    weights: Optional[pd.DataFrame] = None,
    folds: Optional[pd.DataFrame] = None,
    returns: Optional[pd.Series] = None,
    target: float = 1000.0,
    initial: float = 100.0,
    title: str = "Trading agent",
    save_path: Optional[str] = None,
):
    """Stacked figure: equity, drawdown, exposure, per-fold bars, bootstrap fan."""
    import matplotlib.pyplot as plt

    panels = 2 + int(weights is not None) + int(folds is not None) + int(returns is not None)
    heights = [5.0, 2.4] + [2.4] * int(weights is not None) + [3.2] * int(folds is not None) + [
        4.0
    ] * int(returns is not None)
    fig, axes = plt.subplots(panels, 1, figsize=(11, sum(heights)), height_ratios=heights)
    fig.patch.set_facecolor(C_SURFACE)
    axes = np.atleast_1d(axes)

    i = 0
    plot_equity(equity, benchmark, target=target, initial=initial, title=title, ax=axes[i]); i += 1
    plot_drawdown(equity, ax=axes[i]); i += 1
    if weights is not None:
        plot_exposure(weights, ax=axes[i]); i += 1
    if folds is not None:
        plot_folds(folds, ax=axes[i]); i += 1
    if returns is not None:
        plot_monte_carlo(returns, initial=initial, target=target, ax=axes[i]); i += 1

    fig.tight_layout(h_pad=2.0)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=140, facecolor=C_SURFACE, bbox_inches="tight")
    return fig


def fold_table(folds: pd.DataFrame) -> pd.DataFrame:
    """Per-fold results, formatted for reading rather than for machines."""
    out = folds[
        ["fold", "test_start", "test_end", "start_equity", "end_equity", "return", "max_drawdown", "trades"]
    ].copy()
    out["test_start"] = out["test_start"].dt.date
    out["test_end"] = out["test_end"].dt.date
    out["start_equity"] = out["start_equity"].map(lambda v: f"${v:,.2f}")
    out["end_equity"] = out["end_equity"].map(lambda v: f"${v:,.2f}")
    out["return"] = out["return"].map(lambda v: f"{v:+.1%}")
    out["max_drawdown"] = out["max_drawdown"].map(lambda v: f"{v:.1%}")
    return out


def param_frequency(chosen: Sequence[Sequence[Dict]]) -> pd.DataFrame:
    """How often each parameter value was picked across folds.

    A parameter whose winning value jumps around every fold is noise the
    optimiser is chasing; one that is picked consistently is closer to a real
    property of the market.
    """
    rows = []
    for fold_id, picks in enumerate(chosen):
        for rank, params in enumerate(picks):
            for k, v in params.items():
                rows.append({"fold": fold_id, "rank": rank, "param": k, "value": str(v)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    counts = df[df["rank"] == 0].groupby(["param", "value"]).size().rename("times_chosen")
    return counts.reset_index().sort_values(["param", "times_chosen"], ascending=[True, False])
