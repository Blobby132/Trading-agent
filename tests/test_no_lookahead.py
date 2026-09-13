"""End-to-end look-ahead audit by future poisoning.

The truncation tests elsewhere hide the future by shortening the series. This
file does something stricter: it keeps the series exactly as long and replaces
every value after a cut date with **garbage** - a different price level, a
different volatility, a different sign of drift.

The two catch different bugs. Truncation catches code that indexes forward.
Poisoning also catches code whose output *depends on the values* of future bars
without depending on their count: a full-sample mean, a standardisation over
the whole history, a quantile computed once and applied everywhere. Those pass a
truncation test that compares equal-length prefixes and fail this one.

Every assertion is exact equality, not a tolerance. A causal function given the
same past must produce bit-identical output regardless of what happens later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagent import indicators as ind
from tradingagent.agent import AgentConfig, TradingAgent
from tradingagent.cross_section import (
    EqualBlendRanker,
    PortfolioRules,
    RidgeRanker,
    SingleFeatureRanker,
    scores_to_weights,
    volatility_target,
)
from tradingagent.data import synthetic_ohlcv
from tradingagent.engine import BacktestEngine, ExecutionConfig
from tradingagent.features import DEFAULT_FEATURES, feature_panel, raw_features
from tradingagent.optimize import WalkForwardConfig, walk_forward
from tradingagent.risk import RiskConfig, apply_sizing
from tradingagent.strategies import STRATEGY_REGISTRY, make_strategy
from tradingagent.universe import Panel
from tradingagent.xs_optimize import XSWalkForwardConfig, walk_forward_xs

CUT = 600


def poison_frame(df: pd.DataFrame, cut: int = CUT, seed: int = 0) -> pd.DataFrame:
    """Replace every bar after ``cut`` with data from a different world."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out) - cut
    # a different price level, a different vol, the opposite drift
    shock = 500.0 * np.exp(np.cumsum(rng.normal(-0.01, 0.15, size=n)))
    out.iloc[cut:, out.columns.get_loc("close")] = shock
    out.iloc[cut:, out.columns.get_loc("open")] = shock * 1.01
    out.iloc[cut:, out.columns.get_loc("high")] = shock * 1.05
    out.iloc[cut:, out.columns.get_loc("low")] = shock * 0.95
    out.iloc[cut:, out.columns.get_loc("volume")] = rng.lognormal(14.0, 1.0, size=n)
    return out


def poison_panel(panel: Panel, cut: int = CUT, seed: int = 0) -> Panel:
    frames = {
        sym: poison_frame(df, cut, seed + i) for i, (sym, df) in enumerate(panel.to_frames().items())
    }
    return Panel.from_frames(frames)


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return synthetic_ohlcv(900, seed=5)


@pytest.fixture(scope="module")
def dirty(clean) -> pd.DataFrame:
    return poison_frame(clean)


@pytest.fixture(scope="module")
def clean_panel() -> Panel:
    return Panel.from_frames({f"S{i}": synthetic_ohlcv(900, seed=200 + i) for i in range(20)})


@pytest.fixture(scope="module")
def dirty_panel(clean_panel) -> Panel:
    return poison_panel(clean_panel)


def assert_past_identical(a, b, cut: int = CUT, label: str = ""):
    """The first ``cut`` rows must match exactly."""
    left, right = a.iloc[:cut], b.iloc[:cut]
    if isinstance(left, pd.Series):
        pd.testing.assert_series_equal(left, right, check_names=False, rtol=0, atol=0,
                                       obj=label or "series")
    else:
        pd.testing.assert_frame_equal(left, right, rtol=0, atol=0, obj=label or "frame")


# --------------------------------------------------------------------------- #
# layer 1: indicators
# --------------------------------------------------------------------------- #
INDICATORS = {
    "sma": lambda d: ind.sma(d["close"], 20),
    "ema": lambda d: ind.ema(d["close"], 20),
    "rsi": lambda d: ind.rsi(d["close"], 14),
    "atr": lambda d: ind.atr(d["high"], d["low"], d["close"], 14),
    "adx": lambda d: ind.adx(d["high"], d["low"], d["close"], 14),
    "zscore": lambda d: ind.zscore(d["close"], 20),
    "realized_vol": lambda d: ind.realized_vol(d["close"], 20),
    "donchian": lambda d: ind.donchian(d["high"], d["low"], 20)["upper"],
    "bollinger": lambda d: ind.bollinger(d["close"], 20)["upper"],
    "macd": lambda d: ind.macd(d["close"])["macd"],
    "percentile_rank": lambda d: ind.percentile_rank(d["close"], 50),
    "rolling_sharpe": lambda d: ind.rolling_sharpe(d["close"].pct_change(), 30, 252.0),
}


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_ignores_a_poisoned_future(clean, dirty, name):
    fn = INDICATORS[name]
    assert_past_identical(fn(clean), fn(dirty), label=name)


# --------------------------------------------------------------------------- #
# layer 2: single-asset strategies and the ensemble
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(STRATEGY_REGISTRY))
def test_strategy_ignores_a_poisoned_future(clean, dirty, name):
    strat = make_strategy(name)
    assert_past_identical(strat.target_weight(clean), strat.target_weight(dirty), label=name)


def test_ensemble_signal_ignores_a_poisoned_future(clean, dirty):
    """The adaptive blend scores strategies on trailing P&L - trailing only."""
    a, b = TradingAgent(AgentConfig()), TradingAgent(AgentConfig())
    assert_past_identical(a.signal(clean), b.signal(dirty), label="agent signal")


def test_blend_weights_ignore_a_poisoned_future(clean, dirty):
    a, b = TradingAgent(AgentConfig()), TradingAgent(AgentConfig())
    a.signal(clean)
    b.signal(dirty)
    assert_past_identical(a.diagnostics_["blend_weights"], b.diagnostics_["blend_weights"],
                          label="blend weights")


def test_risk_sizing_ignores_a_poisoned_future(clean, dirty):
    cfg = RiskConfig(target_vol=0.5, max_leverage=2.0, kelly_lookback=60)
    a = TradingAgent(AgentConfig(), cfg)
    b = TradingAgent(AgentConfig(), cfg)
    assert_past_identical(a.sized_weight(clean), b.sized_weight(dirty), label="sized weight")


# --------------------------------------------------------------------------- #
# layer 3: cross-sectional features
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", DEFAULT_FEATURES)
def test_raw_feature_ignores_a_poisoned_future(clean_panel, dirty_panel, name):
    assert_past_identical(raw_features(clean_panel)[name], raw_features(dirty_panel)[name],
                          label=name)


@pytest.mark.parametrize("name", DEFAULT_FEATURES)
def test_standardised_feature_ignores_a_poisoned_future(clean_panel, dirty_panel, name):
    """Cross-sectional standardisation is per-date, so poisoning later dates is inert."""
    assert_past_identical(feature_panel(clean_panel)[name], feature_panel(dirty_panel)[name],
                          label=name)


# --------------------------------------------------------------------------- #
# layer 4: rankers and portfolio construction
# --------------------------------------------------------------------------- #
def test_unfitted_rankers_ignore_a_poisoned_future(clean_panel, dirty_panel):
    for ranker in (SingleFeatureRanker("mom_12_1"), EqualBlendRanker()):
        assert_past_identical(
            ranker.score(feature_panel(clean_panel)),
            ranker.score(feature_panel(dirty_panel)),
            label=ranker.name,
        )


def test_ridge_fitted_before_the_cut_ignores_a_poisoned_future(clean_panel, dirty_panel):
    """Coefficients fitted on the past must not move when the future changes."""
    from tradingagent.features import forward_return

    train_start = clean_panel.index[0]
    train_end = clean_panel.index[CUT - 1]

    def fit(panel):
        feats = feature_panel(panel)
        target = forward_return(panel.close, 21)
        model = RidgeRanker(alpha=10.0)
        model.fit(feats, target, train_start=train_start, train_end=train_end, horizon=21)
        return model.coefficients()

    clean_coef, dirty_coef = fit(clean_panel), fit(dirty_panel)
    assert clean_coef.keys() == dirty_coef.keys()
    for key in clean_coef:
        assert clean_coef[key] == dirty_coef[key], (
            f"ridge coefficient {key!r} changed when only the future changed - "
            "the purge is not removing overlapping targets"
        )


def test_scores_to_weights_ignores_a_poisoned_future(clean_panel, dirty_panel):
    rules = PortfolioRules(long_only=True, top_frac=0.2, rebalance_every=5)
    a = scores_to_weights(EqualBlendRanker().score(feature_panel(clean_panel)), rules)
    b = scores_to_weights(EqualBlendRanker().score(feature_panel(dirty_panel)), rules)
    assert_past_identical(a, b, label="weights")


def test_volatility_target_ignores_a_poisoned_future(clean_panel, dirty_panel):
    rules = PortfolioRules(long_only=True, rebalance_every=5)
    def sized(panel):
        w = scores_to_weights(EqualBlendRanker().score(feature_panel(panel)), rules)
        return volatility_target(w, panel.close, target_vol=0.15, periods_per_year=252.0)
    assert_past_identical(sized(clean_panel), sized(dirty_panel), label="vol-targeted weights")


# --------------------------------------------------------------------------- #
# layer 5: the execution engine
# --------------------------------------------------------------------------- #
def test_engine_equity_ignores_a_poisoned_future(clean, dirty):
    """Fills, costs and stops up to the cut cannot depend on later prices."""
    weights = pd.Series(np.tile([1.0, 0.0, -0.5], len(clean))[: len(clean)], index=clean.index)
    cfg = ExecutionConfig(initial_capital=100.0, min_trade_frac=0.0)
    risk = RiskConfig(target_vol=0.0, atr_stop_mult=4.0, max_drawdown_stop=0.3)
    a = BacktestEngine(cfg, risk).run(clean, weights)
    b = BacktestEngine(cfg, risk).run(dirty, weights)
    assert_past_identical(a.equity, b.equity, label="equity")
    assert_past_identical(a.weights, b.weights, label="positions")


def test_engine_trade_log_ignores_a_poisoned_future(clean, dirty):
    weights = pd.Series(np.tile([1.0, -1.0], len(clean))[: len(clean)], index=clean.index)
    cfg = ExecutionConfig(initial_capital=100.0, min_trade_frac=0.0)
    risk = RiskConfig(target_vol=0.0, atr_stop_mult=3.0, max_drawdown_stop=0.0)
    cut_ts = clean.index[CUT - 1]
    a = BacktestEngine(cfg, risk).run(clean, weights).trades.loc[:cut_ts]
    b = BacktestEngine(cfg, risk).run(dirty, weights).trades.loc[:cut_ts]
    pd.testing.assert_frame_equal(a, b, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# layer 6: the walk-forward loops, end to end
# --------------------------------------------------------------------------- #
def test_single_asset_walk_forward_ignores_a_poisoned_future():
    """Folds that finish before the cut must be untouched by what comes after."""
    clean = synthetic_ohlcv(1400, seed=9)
    dirty = poison_frame(clean, cut=900, seed=3)
    cfg = WalkForwardConfig(train_bars=400, test_bars=150, n_candidates=6, top_k=2,
                            seed=0, verbose=False)
    a = walk_forward(clean, ExecutionConfig(initial_capital=100.0), cfg)
    b = walk_forward(dirty, ExecutionConfig(initial_capital=100.0), cfg)

    cut_ts = clean.index[900]
    early = a.folds[a.folds["test_end"] < cut_ts]
    assert len(early) >= 2, "test needs at least two folds before the cut"
    for col in ("start_equity", "end_equity", "return"):
        np.testing.assert_array_equal(
            early[col].to_numpy(), b.folds.loc[early.index, col].to_numpy()
        )


def test_cross_sectional_walk_forward_ignores_a_poisoned_future():
    clean = Panel.from_frames({f"S{i}": synthetic_ohlcv(1400, seed=300 + i) for i in range(20)})
    dirty = poison_panel(clean, cut=900, seed=7)
    cfg = XSWalkForwardConfig(train_bars=500, test_bars=150, embargo_bars=10, n_candidates=4,
                              top_k=2, seed=0, verbose=False)
    exec_cfg = ExecutionConfig(initial_capital=100.0, periods_per_year=252.0)
    a = walk_forward_xs(clean, exec_cfg, cfg)
    b = walk_forward_xs(dirty, exec_cfg, cfg)

    cut_ts = clean.index[900]
    early = a.folds[a.folds["test_end"] < cut_ts]
    assert len(early) >= 1, "test needs at least one fold before the cut"
    for col in ("start_equity", "end_equity", "return"):
        np.testing.assert_array_equal(
            early[col].to_numpy(), b.folds.loc[early.index, col].to_numpy()
        )
    # and the model selected on those folds must be the same model
    np.testing.assert_array_equal(
        early["model"].to_numpy(), b.folds.loc[early.index, "model"].to_numpy()
    )


# --------------------------------------------------------------------------- #
# a control: the test itself must be capable of failing
# --------------------------------------------------------------------------- #
def test_the_audit_catches_a_deliberately_leaky_indicator(clean, dirty):
    """A negative control - if this passes, the poisoning is not biting."""
    def leaky(df):
        # full-sample normalisation: a classic, and invisible to a length-based test
        return (df["close"] - df["close"].mean()) / df["close"].std()

    with pytest.raises(AssertionError):
        assert_past_identical(leaky(clean), leaky(dirty), label="leaky")


def test_the_audit_catches_a_forward_shifted_signal(clean, dirty):
    def peeking(df):
        return df["close"].shift(-1) / df["close"] - 1.0

    with pytest.raises(AssertionError):
        assert_past_identical(peeking(clean), peeking(dirty), label="peeking")


def test_an_unpurged_fit_would_leak_the_poisoned_future(clean_panel, dirty_panel):
    """Negative control for the purge specifically.

    The ridge target is a 21-bar forward return. Training rows in the last 21
    bars of the window resolve *after* it, so without purging their targets are
    read from the poisoned future and the fitted coefficients move. This proves
    the purge test above is not passing vacuously.
    """
    from tradingagent.features import forward_return, stack

    train_end = clean_panel.index[CUT - 1]

    def fit_without_purge(panel):
        feats = {k: v.loc[: train_end] for k, v in feature_panel(panel).items()}
        target = forward_return(panel.close, 21).loc[: train_end]
        X, y, _, names = stack(feats, target)          # no purge_overlapping call
        return RidgeRanker(alpha=10.0).fit_matrix(X, y, names).coefficients()

    leaky_clean = fit_without_purge(clean_panel)
    leaky_dirty = fit_without_purge(dirty_panel)
    assert any(
        leaky_clean[k] != leaky_dirty[k] for k in leaky_clean
    ), "poisoning did not reach the unpurged targets - the purge test proves nothing"
