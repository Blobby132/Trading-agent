import numpy as np
import pandas as pd
import pytest

from tradingagent.cross_section import (
    DEFAULT_SIGNS,
    EqualBlendRanker,
    PortfolioRules,
    RidgeRanker,
    SingleFeatureRanker,
    make_ranker,
    scores_to_weights,
    volatility_target,
)


@pytest.fixture
def scores():
    idx = pd.date_range("2021-01-01", periods=60, freq="1D", tz="UTC")
    cols = [f"S{i}" for i in range(20)]
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(60, 20)), index=idx, columns=cols)


def test_weights_respect_the_gross_budget(scores):
    w = scores_to_weights(scores, PortfolioRules(long_only=True, gross=1.0, rebalance_every=1))
    gross = w.abs().sum(axis=1)
    assert np.allclose(gross[gross > 0], 1.0, atol=1e-9)


def test_long_only_never_shorts(scores):
    w = scores_to_weights(scores, PortfolioRules(long_only=True, rebalance_every=1))
    assert (w >= -1e-12).all().all()


def test_long_short_is_balanced(scores):
    w = scores_to_weights(scores, PortfolioRules(long_only=False, gross=1.0, rebalance_every=1))
    net = w.sum(axis=1)
    assert np.allclose(net[w.abs().sum(axis=1) > 0], 0.0, atol=1e-9)


def test_per_name_cap_is_enforced(scores):
    w = scores_to_weights(scores, PortfolioRules(max_weight=0.1, rebalance_every=1))
    assert w.abs().max().max() <= 0.1 + 1e-9


def test_thin_universe_never_concentrates_into_two_names():
    """The failure this rule exists to prevent: a 'cross-section' of 2 names."""
    idx = pd.date_range("2021-01-01", periods=30, freq="1D", tz="UTC")
    thin = pd.DataFrame(np.random.default_rng(1).normal(size=(30, 14)), index=idx,
                        columns=[f"S{i}" for i in range(14)])
    rules = PortfolioRules(long_only=True, top_frac=0.1, min_positions=5, min_names=12,
                           rebalance_every=1)
    w = scores_to_weights(thin, rules)
    held = (w.abs() > 1e-9).sum(axis=1)
    assert held[held > 0].min() >= 5


def test_universe_below_min_names_stands_aside():
    idx = pd.date_range("2021-01-01", periods=30, freq="1D", tz="UTC")
    tiny = pd.DataFrame(np.random.default_rng(2).normal(size=(30, 4)), index=idx,
                        columns=list("ABCD"))
    w = scores_to_weights(tiny, PortfolioRules(min_names=12, rebalance_every=1))
    assert (w == 0).all().all()


def test_rebalance_schedule_reduces_turnover(scores):
    daily = scores_to_weights(scores, PortfolioRules(rebalance_every=1))
    weekly = scores_to_weights(scores, PortfolioRules(rebalance_every=5))
    assert weekly.diff().abs().sum().sum() < daily.diff().abs().sum().sum()


def test_ranker_registry_builds_each_kind():
    for spec in ("ridge", "equal_blend", "mom_only", "rev_only"):
        assert make_ranker(spec, ridge_alpha=10.0) is not None
    with pytest.raises(KeyError):
        make_ranker("nope")


def test_ridge_recovers_a_planted_signal():
    """If one feature genuinely predicts, ridge must put weight on it."""
    rng = np.random.default_rng(3)
    n, k = 4000, 5
    X = rng.normal(size=(n, k))
    y = 0.4 * X[:, 2] + rng.normal(scale=1.0, size=n)     # only feature 2 matters
    names = [f"f{i}" for i in range(k)]
    model = RidgeRanker(alpha=1.0).fit_matrix(X, y, names)
    coef = model.coefficients()
    assert coef["f2"] == max(coef.values())
    assert coef["f2"] > 2 * max(abs(v) for kk, v in coef.items() if kk != "f2")


def test_ridge_shrinks_toward_zero_as_alpha_grows():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(2000, 4))
    y = 0.3 * X[:, 0] + rng.normal(size=2000)
    names = [f"f{i}" for i in range(4)]
    light = RidgeRanker(alpha=0.01).fit_matrix(X, y, names).coefficients()
    heavy = RidgeRanker(alpha=500.0).fit_matrix(X, y, names).coefficients()
    assert abs(heavy["f0"]) < abs(light["f0"])


def test_ridge_refuses_to_score_before_fitting():
    with pytest.raises(RuntimeError):
        RidgeRanker().score({"mom_3": pd.DataFrame()})


def test_ridge_with_too_few_observations_returns_zero_coefficients():
    X = np.random.default_rng(5).normal(size=(10, 3))
    y = np.random.default_rng(6).normal(size=10)
    coef = RidgeRanker().fit_matrix(X, y, ["a", "b", "c"]).coefficients()
    assert all(v == 0.0 for v in coef.values())


def test_volatility_target_scales_the_book_down_when_vol_is_high():
    idx = pd.date_range("2021-01-01", periods=400, freq="1D", tz="UTC")
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.05, size=(400, 6)), axis=0)),
        index=idx, columns=[f"S{i}" for i in range(6)],
    )
    w = pd.DataFrame(1.0 / 6, index=idx, columns=close.columns)
    scaled = volatility_target(w, close, target_vol=0.10, periods_per_year=252.0)
    # 5% daily vol is ~79% annualised, far above a 10% target
    assert scaled.abs().sum(axis=1).tail(200).mean() < w.abs().sum(axis=1).tail(200).mean()


def test_cap_wins_over_the_gross_budget_in_a_thin_universe():
    """When the cap cannot fund the target gross, the book runs smaller."""
    idx = pd.date_range("2021-01-01", periods=20, freq="1D", tz="UTC")
    thin = pd.DataFrame(np.random.default_rng(8).normal(size=(20, 14)), index=idx,
                        columns=[f"S{i}" for i in range(14)])
    rules = PortfolioRules(long_only=True, gross=1.0, max_weight=0.1, min_names=12,
                           rebalance_every=1)
    w = scores_to_weights(thin, rules)
    assert w.abs().max().max() <= 0.1 + 1e-9
    # 14 names, at most 7 per side, 0.1 each -> 0.7 gross, not the requested 1.0
    assert w.abs().sum(axis=1).max() <= 0.75
