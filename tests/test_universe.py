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


def test_load_panel_gives_up_when_the_source_is_down(monkeypatch):
    """A dead source must fail fast, not retry a hundred tickers."""
    import tradingagent.universe as uni

    attempts = []

    def always_fails(symbol, **kwargs):
        attempts.append(symbol)
        raise RuntimeError("rate limited")

    monkeypatch.setattr(uni, "load_prices", always_fails)
    with pytest.raises(RuntimeError, match="failed on 3 symbols in a row"):
        uni.load_panel([f"T{i}" for i in range(50)], source="nasdaq", pause=0.0,
                       max_consecutive_failures=3, verbose=False)
    assert len(attempts) == 3, "kept going after the source was clearly unavailable"


def test_load_panel_tolerates_isolated_failures(monkeypatch):
    """One bad ticker among good ones must not stop the run."""
    import tradingagent.universe as uni
    from tradingagent.data import synthetic_ohlcv

    def sometimes_fails(symbol, **kwargs):
        if symbol in {"BAD1", "BAD2"}:
            raise RuntimeError("delisted")
        return synthetic_ohlcv(400, seed=abs(hash(symbol)) % 1000)

    monkeypatch.setattr(uni, "load_prices", sometimes_fails)
    panel = uni.load_panel(["A", "BAD1", "B", "BAD2", "C", "D"], source="nasdaq", pause=0.0,
                           max_consecutive_failures=3, min_bars=100, verbose=False)
    assert set(panel.symbols) == {"A", "B", "C", "D"}


def test_require_history_before_is_point_in_time():
    """Membership must be decided on data available at the cutoff, not later."""
    from tradingagent.data import synthetic_ohlcv

    early = synthetic_ohlcv(600, seed=1, start="2018-01-01")
    late = synthetic_ohlcv(600, seed=2, start="2019-06-01")   # lists 18 months later
    panel = Panel.from_frames({"EARLY": early, "LATE": late})

    # by the whole-sample filter both qualify, because both end up with 600 bars
    assert set(panel.min_history(500).symbols) == {"EARLY", "LATE"}
    # but at the start of 2019 only one of them had any history at all
    pit = panel.require_history_before("2019-01-01", 200)
    assert set(pit.symbols) == {"EARLY"}, "a name that had not listed yet was selected"


def test_require_history_before_keeps_every_field_aligned():
    from tradingagent.data import synthetic_ohlcv

    panel = Panel.from_frames({f"S{i}": synthetic_ohlcv(400, seed=i) for i in range(5)})
    kept = panel.require_history_before(panel.index[300], 100)
    assert len(kept.symbols) == 5
    for field in ("open", "high", "low", "close", "volume"):
        assert list(getattr(kept, field).columns) == kept.symbols


# --------------------------------------------------------------------------- #
# point-in-time membership
# --------------------------------------------------------------------------- #
def test_static_membership_admits_it_is_biased():
    from tradingagent.universe import StaticMembership

    provider = StaticMembership(["A", "B"])
    assert provider.point_in_time is False
    assert "NOT point-in-time" in provider.describe()


def test_point_in_time_membership_respects_entry_and_exit(tmp_path):
    from tradingagent.universe import PointInTimeMembership

    path = tmp_path / "members.csv"
    path.write_text("symbol,entered,exited\nA,2019-01-01,\nB,2020-06-01,\nC,2019-01-01,2020-04-01\n")
    provider = PointInTimeMembership.from_csv(str(path))
    assert provider.point_in_time is True
    assert set(provider.members_on(pd.Timestamp("2019-06-01", tz="UTC"))) == {"A", "C"}
    assert set(provider.members_on(pd.Timestamp("2020-07-01", tz="UTC"))) == {"A", "B"}


def test_membership_file_requires_the_right_columns(tmp_path):
    from tradingagent.universe import PointInTimeMembership

    path = tmp_path / "bad.csv"
    path.write_text("ticker,date\nA,2019-01-01\n")
    with pytest.raises(ValueError, match="needs columns"):
        PointInTimeMembership.from_csv(str(path))


def test_applying_membership_blanks_non_members(tmp_path):
    from tradingagent.universe import PointInTimeMembership, apply_membership

    panel = Panel.from_frames({
        s: synthetic_ohlcv(400, seed=i, start="2020-01-01")
        for i, s in enumerate(["A", "B", "C"])
    })
    path = tmp_path / "m.csv"
    path.write_text("symbol,entered,exited\nA,2019-01-01,\nB,2020-06-01,\nC,2019-01-01,2020-04-01\n")
    restricted = apply_membership(panel, PointInTimeMembership.from_csv(str(path)))

    assert restricted.close["B"].loc[:"2020-05-30"].isna().all(), "traded before it was a member"
    assert restricted.close["C"].loc["2020-05-01":].isna().all(), "traded after it left"
    assert restricted.close["A"].notna().any()
    # the engine needs no change: a non-member is simply not tradeable
    assert not restricted.tradeable()["B"].loc[:"2020-05-30"].any()


def test_static_membership_leaves_everything_tradeable():
    from tradingagent.universe import StaticMembership, apply_membership

    panel = Panel.from_frames({s: synthetic_ohlcv(200, seed=i) for i, s in enumerate(["A", "B"])})
    same = apply_membership(panel, StaticMembership(["A", "B"]))
    pd.testing.assert_frame_equal(same.close, panel.close)
