import numpy as np
import pandas as pd
import pytest

from tradingagent.features import (
    DEFAULT_FEATURES,
    cross_sectional_z,
    feature_panel,
    forward_return,
    purge_overlapping,
    raw_features,
    stack,
)
from tradingagent.universe import Panel


@pytest.fixture(scope="module")
def panel():
    from tradingagent.data import synthetic_ohlcv

    frames = {f"S{i}": synthetic_ohlcv(900, seed=i, interval="1d") for i in range(12)}
    return Panel.from_frames(frames)


@pytest.mark.parametrize("name", DEFAULT_FEATURES)
def test_raw_feature_is_causal(panel, name):
    """Truncating the future must not change any past feature value."""
    cut = 700
    full = raw_features(panel)[name].iloc[:cut]
    truncated = raw_features(panel.slice(slice(0, cut)))[name]
    pd.testing.assert_frame_equal(full, truncated, rtol=1e-9, atol=1e-9)


def test_cross_sectional_z_is_standardised():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(5.0, 3.0, size=(50, 20)))
    z = cross_sectional_z(frame)
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(z.std(axis=1, ddof=0), 1.0, atol=1e-9)


def test_cross_sectional_z_ignores_missing_names():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, np.nan], "c": [3.0, 4.0],
                          "d": [4.0, 5.0], "e": [5.0, 6.0], "f": [6.0, 7.0]})
    z = cross_sectional_z(frame, min_names=3)
    assert z.notna().sum(axis=1).tolist() == [6, 5]
    assert np.isnan(z.loc[1, "b"])


def test_cross_sectional_z_stands_down_on_a_thin_row():
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
    assert cross_sectional_z(frame, min_names=5).isna().all().all()


def test_forward_return_looks_forward_and_is_demeaned():
    idx = pd.date_range("2021-01-01", periods=10, freq="1D", tz="UTC")
    close = pd.DataFrame({"a": np.arange(1, 11.0), "b": np.arange(10, 0, -1.0)}, index=idx)
    raw = forward_return(close, 1, demean=False)
    assert raw.loc[idx[0], "a"] == pytest.approx(1.0)       # 1 -> 2
    assert np.isnan(raw.loc[idx[-1], "a"])                  # no future left
    demeaned = forward_return(close, 1, demean=True)
    assert np.allclose(demeaned.dropna(how="all").sum(axis=1), 0.0, atol=1e-12)


def test_feature_panel_only_scores_tradeable_bars(panel):
    holed = Panel(
        panel.open.copy(), panel.high.copy(), panel.low.copy(), panel.close.copy(), panel.volume.copy()
    )
    holed.close.iloc[:100, 0] = np.nan
    holed.open.iloc[:100, 0] = np.nan
    feats = feature_panel(holed)
    assert feats["mom_3"].iloc[:100, 0].isna().all()


def test_stack_drops_incomplete_rows():
    idx = pd.date_range("2021-01-01", periods=3, freq="1D", tz="UTC")
    f1 = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [1.0, 2.0, 3.0]}, index=idx)
    f2 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, 2.0, 3.0]}, index=idx)
    y = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.1, 0.2, 0.3]}, index=idx)
    X, target, index, names = stack({"f1": f1, "f2": f2}, y)
    assert X.shape == (4, 2)          # 6 cells minus 2 with a missing feature
    assert len(target) == 4
    assert names == ["f1", "f2"]


def test_purge_removes_rows_whose_outcome_lands_in_the_test_window():
    dates = pd.date_range("2021-01-01", periods=100, freq="1D", tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["a", "b"]])
    train_end = dates[59]
    keep = purge_overlapping(index, train_end, horizon=10, dates=dates)
    kept_dates = index.get_level_values(0)[keep]
    # a 10-day forward return observed on day 49 resolves on day 59, which is
    # train_end itself - the last observation whose outcome is fully inside the
    # training window. Day 50 would resolve on day 60, in the test period.
    assert kept_dates.max() == dates[49]
    assert not keep.all()


def test_purge_with_a_horizon_longer_than_the_window_keeps_nothing():
    dates = pd.date_range("2021-01-01", periods=20, freq="1D", tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["a"]])
    keep = purge_overlapping(index, dates[5], horizon=50, dates=dates)
    assert not keep.any()
