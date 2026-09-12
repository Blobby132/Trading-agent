import numpy as np
import pandas as pd
import pytest

from tradingagent import indicators as ind

INDICATOR_CALLS = {
    "sma": lambda d: ind.sma(d["close"], 20),
    "ema": lambda d: ind.ema(d["close"], 20),
    "rsi": lambda d: ind.rsi(d["close"], 14),
    "atr": lambda d: ind.atr(d["high"], d["low"], d["close"], 14),
    "adx": lambda d: ind.adx(d["high"], d["low"], d["close"], 14),
    "zscore": lambda d: ind.zscore(d["close"], 20),
    "realized_vol": lambda d: ind.realized_vol(d["close"], 20),
    "donchian_upper": lambda d: ind.donchian(d["high"], d["low"], 20)["upper"],
    "bollinger_upper": lambda d: ind.bollinger(d["close"], 20)["upper"],
    "macd": lambda d: ind.macd(d["close"])["macd"],
    "percentile_rank": lambda d: ind.percentile_rank(d["close"], 50),
}


@pytest.mark.parametrize("name", sorted(INDICATOR_CALLS))
def test_indicator_is_causal(prices, name):
    """Truncating the future must not change any past value.

    This is the single most important property in the whole package: an
    indicator that fails it turns a backtest into a look at the answer sheet.
    """
    fn = INDICATOR_CALLS[name]
    cut = 600
    full = fn(prices).iloc[:cut]
    truncated = fn(prices.iloc[:cut])
    pd.testing.assert_series_equal(full, truncated, check_names=False, rtol=1e-12, atol=1e-12)


def test_donchian_excludes_current_bar(prices):
    """The channel must be built from completed bars only."""
    ch = ind.donchian(prices["high"], prices["low"], 10)
    manual_upper = prices["high"].shift(1).rolling(10).max()
    pd.testing.assert_series_equal(ch["upper"], manual_upper, check_names=False)


def test_atr_matches_manual_wilder():
    idx = pd.date_range("2021-01-01", periods=5, freq="1D", tz="UTC")
    high = pd.Series([10.0, 11.0, 12.0, 11.5, 13.0], index=idx)
    low = pd.Series([9.0, 10.0, 11.0, 10.0, 12.0], index=idx)
    close = pd.Series([9.5, 10.5, 11.5, 10.5, 12.5], index=idx)
    tr = ind.true_range(high, low, close)
    # first bar: high - low; later bars include the gap against the prior close
    assert tr.iloc[0] == pytest.approx(1.0)
    assert tr.iloc[1] == pytest.approx(1.5)
    assert ind.atr(high, low, close, 2).notna().sum() == 4


def test_rsi_bounds(prices):
    r = ind.rsi(prices["close"], 14).dropna()
    assert r.between(0.0, 100.0).all()


def test_rolling_drawdown_is_non_positive(prices):
    eq = (1.0 + prices["close"].pct_change().fillna(0.0)).cumprod()
    assert (ind.rolling_drawdown(eq) <= 1e-12).all()
