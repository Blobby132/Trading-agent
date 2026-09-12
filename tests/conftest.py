import numpy as np
import pandas as pd
import pytest

from tradingagent.data import synthetic_ohlcv


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    return synthetic_ohlcv(900, seed=42, interval="1d")


@pytest.fixture(scope="session")
def trending_prices() -> pd.DataFrame:
    """A clean uptrend, so directional logic has something unambiguous to find."""
    idx = pd.date_range("2020-01-01", periods=400, freq="1D", tz="UTC")
    close = pd.Series(100.0 * (1.004 ** np.arange(400)), index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(100.0),
            "high": close * 1.004,
            "low": close * 0.997,
            "close": close,
            "volume": 1000.0,
        }
    )
