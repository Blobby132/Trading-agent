import numpy as np
import pandas as pd
import pytest

from tradingagent.data import (
    OHLCV_COLUMNS,
    align_universe,
    bars_per_year,
    load_coinbase,
    synthetic_ohlcv,
)


def test_synthetic_frame_is_well_formed():
    df = synthetic_ohlcv(300, seed=1)
    assert list(df.columns) == OHLCV_COLUMNS
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert str(df.index.tz) == "UTC"
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["close"] > 0).all()


def test_synthetic_data_is_reproducible():
    a = synthetic_ohlcv(200, seed=99)
    b = synthetic_ohlcv(200, seed=99)
    pd.testing.assert_frame_equal(a, b)
    assert not synthetic_ohlcv(200, seed=100)["close"].equals(a["close"])


def test_bars_per_year_known_intervals():
    assert bars_per_year("1d") == 365
    assert bars_per_year("1h") == 365 * 24
    with pytest.raises(KeyError):
        bars_per_year("1fortnight")


def test_align_universe_produces_one_shared_index():
    a = synthetic_ohlcv(300, seed=1)
    b = synthetic_ohlcv(250, seed=2)
    aligned = align_universe({"a": a, "b": b})
    idx = aligned["a"].index
    assert all(df.index.equals(idx) for df in aligned.values())


def test_default_end_date_does_not_raise(monkeypatch):
    """A missing `end` must default to 'now' without a timezone error.

    pandas returns a tz-aware timestamp from utcnow(), so localising it again
    raises - which broke every call that did not pass an explicit end date.
    """
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return []

    class FakeSession:
        def get(self, url, params=None, timeout=None):
            captured["params"] = params
            return FakeResponse()

    with pytest.raises(RuntimeError, match="no candles"):
        load_coinbase("BTC-USD", interval="1d", start="2024-01-01", end=None,
                      session=FakeSession(), pause=0.0)
    assert "start" in captured["params"] and "end" in captured["params"]
