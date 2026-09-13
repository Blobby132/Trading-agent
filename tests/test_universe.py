import numpy as np
import pandas as pd
import pytest

from tradingagent.data import synthetic_ohlcv
from tradingagent.universe import UNIVERSES, US_LARGE_CAP, Panel


@pytest.fixture
def frames():
    out = {f"S{i}": synthetic_ohlcv(300, seed=i) for i in range(4)}
    out["LATE"] = synthetic_ohlcv(120, seed=99, start="2018-06-01")   # lists later
    return out


def test_universes_have_no_duplicates():
    for name, names in UNIVERSES.items():
        assert len(names) == len(set(names)), f"{name} contains duplicate tickers"


def test_panel_uses_the_union_of_dates(frames):
    panel = Panel.from_frames(frames)
    assert len(panel) == max(len(f) for f in frames.values())
    assert set(panel.symbols) == set(frames)


def test_late_listing_is_nan_not_forward_filled(frames):
    panel = Panel.from_frames(frames)
    late = panel.close["LATE"]
    assert late.isna().any(), "a name that had not listed must be NaN"
    first_valid = late.first_valid_index()
    assert late.loc[:first_valid].isna().sum() > 0
    assert not panel.tradeable().loc[: late.index[0], "LATE"].any()


def test_round_trip_through_frames(frames):
    panel = Panel.from_frames(frames)
    back = panel.to_frames()
    assert set(back) == set(frames)
    for sym, df in back.items():
        assert df.index.equals(panel.index)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_min_history_drops_short_series(frames):
    panel = Panel.from_frames(frames).min_history(200)
    assert "LATE" not in panel.symbols
    assert len(panel.symbols) == 4


def test_slice_keeps_every_field_aligned(frames):
    panel = Panel.from_frames(frames).slice(slice(10, 50))
    assert len(panel) == 40
    for field in ("open", "high", "low", "close", "volume"):
        assert len(getattr(panel, field)) == 40


def test_describe_reports_coverage(frames):
    described = Panel.from_frames(frames).describe()
    assert set(described.columns) == {"symbol", "bars", "first", "last"}
    assert described["bars"].is_monotonic_decreasing


def test_large_cap_universe_is_big_enough_to_rank():
    # the whole cross-sectional premise needs breadth; guard against the list
    # being trimmed below the point where ranking means anything
    assert len(US_LARGE_CAP) >= 50
