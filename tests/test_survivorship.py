import numpy as np
import pandas as pd
import pytest

from tradingagent.cross_section import PortfolioRules, SingleFeatureRanker, scores_to_weights
from tradingagent.data import synthetic_ohlcv
from tradingagent.features import feature_panel
from tradingagent.survivorship import compare_universes, inject_failures, stress_test
from tradingagent.universe import Panel


@pytest.fixture(scope="module")
def panel():
    return Panel.from_frames({f"S{i}": synthetic_ohlcv(900, seed=600 + i) for i in range(25)})


def momentum_weights(p: Panel) -> pd.DataFrame:
    feats = feature_panel(p)
    rules = PortfolioRules(long_only=True, top_frac=0.2, gross=1.0, max_weight=0.2,
                           rebalance_every=21, min_names=10)
    return scores_to_weights(SingleFeatureRanker("mom_12_1").score(feats), rules)


def test_no_failures_leaves_the_panel_untouched(panel):
    damaged, events = inject_failures(panel, failure_rate=0.0, seed=0)
    assert events == []
    pd.testing.assert_frame_equal(damaged.close, panel.close)


def test_injected_failures_actually_delist(panel):
    damaged, events = inject_failures(panel, failure_rate=0.15, seed=1, concentration=0.0)
    assert events, "no failures injected at a 15%/yr rate"
    for sym, when in events:
        series = damaged.close[sym]
        assert series.loc[when:].isna().any(), f"{sym} never stopped trading"
        # and it declined into the delisting rather than vanishing at full price
        last_live = series.dropna()
        assert last_live.iloc[-1] < series.loc[:when].dropna().iloc[-1]


def test_failures_are_concentrated_where_they_are_told_to_be(panel):
    """Real failures land on names that have already been falling."""
    feats = feature_panel(panel)
    signal = SingleFeatureRanker("mom_12_1").score(feats)
    avg = signal.mean(axis=0)

    weak_hits, uniform_hits = 0, 0
    for seed in range(12):
        _, weak = inject_failures(panel, failure_rate=0.10, rank_signal=signal,
                                  concentration=6.0, seed=seed)
        _, flat = inject_failures(panel, failure_rate=0.10, concentration=0.0, seed=seed)
        weak_hits += sum(avg.rank(pct=True).get(s, 0.5) < 0.5 for s, _ in weak)
        uniform_hits += sum(avg.rank(pct=True).get(s, 0.5) < 0.5 for s, _ in flat)
    assert weak_hits > uniform_hits, "concentration had no effect on who died"


def test_stress_test_reports_a_row_per_failure_rate(panel):
    out = stress_test(panel, momentum_weights, failure_rates=(0.0, 0.05),
                      n_trials=2, verbose=False)
    assert list(out["failure_rate"]) == [0.0, 0.05]
    assert out.loc[out.failure_rate == 0.0, "strategy_drag"].iloc[0] == pytest.approx(0.0)
    assert np.isfinite(out["relative_drag"]).all()


def test_stress_test_damage_grows_with_the_failure_rate(panel):
    out = stress_test(panel, momentum_weights, failure_rates=(0.0, 0.02, 0.20),
                      n_trials=4, concentration=0.0, verbose=False)
    mild = out.loc[out.failure_rate == 0.02, "benchmark_drag"].iloc[0]
    severe = out.loc[out.failure_rate == 0.20, "benchmark_drag"].iloc[0]
    assert severe < mild, "a 10x higher failure rate did not cost the benchmark more"


def test_compare_universes_measures_the_gap(panel):
    """Survivors-only against survivors-plus-the-dead."""
    damaged, events = inject_failures(panel, failure_rate=0.10, seed=3, concentration=0.0)
    assert events
    survivors = Panel(*(getattr(panel, f)[[s for s in panel.symbols
                                           if s not in {e[0] for e in events}]]
                        for f in ("open", "high", "low", "close", "volume")))
    out = compare_universes(survivors, damaged, momentum_weights)
    assert out["n_survivors"] < out["n_with_delisted"]
    assert np.isfinite(out["survivorship_bias"])
