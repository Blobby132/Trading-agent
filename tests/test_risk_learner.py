import numpy as np
import pandas as pd
import pytest

from tradingagent.data import synthetic_ohlcv
from tradingagent.risk_learner import (
    RiskLearner,
    RiskLearnerConfig,
    correlation_scale,
    drawdown_scale,
    inverse_vol_weights,
    portfolio_vol_scale,
)
from tradingagent.universe import Panel


@pytest.fixture(scope="module")
def panel():
    return Panel.from_frames({f"S{i}": synthetic_ohlcv(800, seed=500 + i) for i in range(12)})


@pytest.fixture
def equal_weights(panel):
    return pd.DataFrame(1.0 / 12, index=panel.index, columns=panel.symbols)


def test_inverse_vol_preserves_gross_exposure(panel, equal_weights):
    """It reallocates risk; it must not quietly change how much is invested."""
    out = inverse_vol_weights(equal_weights, panel.close)
    before = equal_weights.abs().sum(axis=1)
    after = out.abs().sum(axis=1)
    live = after > 0
    np.testing.assert_allclose(after[live], before[live], rtol=1e-9)


def test_inverse_vol_gives_the_calmer_name_more_weight():
    idx = pd.date_range("2020-01-01", periods=400, freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 400)))
    wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.050, 400)))
    close = pd.DataFrame({"CALM": calm, "WILD": wild}, index=idx)
    w = pd.DataFrame(0.5, index=idx, columns=["CALM", "WILD"])
    out = inverse_vol_weights(w, close)
    assert out["CALM"].iloc[-1] > out["WILD"].iloc[-1]


def test_correlation_scale_cuts_an_undiversified_book():
    """Twelve names that move together is one position wearing twelve tickers."""
    idx = pd.date_range("2020-01-01", periods=500, freq="1D", tz="UTC")
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.01, 500)
    identical = pd.DataFrame(
        {f"S{i}": 100 * np.exp(np.cumsum(common)) for i in range(6)}, index=idx
    )
    independent = pd.DataFrame(
        {f"S{i}": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))) for i in range(6)}, index=idx
    )
    w = pd.DataFrame(1.0 / 6, index=idx, columns=[f"S{i}" for i in range(6)])
    tight = correlation_scale(w, identical).tail(200).mean()
    loose = correlation_scale(w, independent).tail(200).mean()
    assert tight < loose


def test_portfolio_vol_scale_targets_the_requested_volatility():
    idx = pd.date_range("2020-01-01", periods=600, freq="1D", tz="UTC")
    rng = np.random.default_rng(2)
    close = pd.DataFrame(
        {f"S{i}": 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 600))) for i in range(4)}, index=idx
    )
    w = pd.DataFrame(0.25, index=idx, columns=close.columns)
    low = portfolio_vol_scale(w, close, target_vol=0.10, periods_per_year=252.0)
    high = portfolio_vol_scale(w, close, target_vol=0.40, periods_per_year=252.0)
    assert low.tail(300).mean() < high.tail(300).mean()
    assert (low <= 2.0 + 1e-9).all()


def test_portfolio_vol_scale_is_inert_when_disabled(panel, equal_weights):
    out = portfolio_vol_scale(equal_weights, panel.close, target_vol=0.0)
    assert (out == 1.0).all()


def test_drawdown_scale_cuts_into_a_drawdown_and_recovers():
    idx = pd.date_range("2020-01-01", periods=300, freq="1D", tz="UTC")
    path = np.concatenate([
        np.linspace(100, 140, 100),    # rally
        np.linspace(140, 70, 100),     # deep drawdown
        np.linspace(70, 150, 100),     # full recovery
    ])
    close = pd.DataFrame({"S": path}, index=idx)
    w = pd.DataFrame(1.0, index=idx, columns=["S"])
    scale = drawdown_scale(w, close, threshold=0.10, floor=0.25)
    assert scale.iloc[:90].mean() == pytest.approx(1.0, abs=1e-9)   # no drawdown yet
    assert scale.iloc[180:200].mean() < 0.6                        # cut at the bottom
    assert scale.iloc[-1] == pytest.approx(1.0, abs=1e-9)          # back to full after recovery
    assert (scale >= 0.25 - 1e-9).all()                            # never below the floor


def test_drawdown_scale_never_latches_off():
    """The scar tissue: an earlier kill switch latched on for good."""
    idx = pd.date_range("2020-01-01", periods=400, freq="1D", tz="UTC")
    path = np.concatenate([np.linspace(100, 50, 200), np.linspace(50, 55, 200)])
    close = pd.DataFrame({"S": path}, index=idx)
    w = pd.DataFrame(1.0, index=idx, columns=["S"])
    scale = drawdown_scale(w, close, threshold=0.10, floor=0.25)
    assert scale.iloc[-1] > 0.2, "scaling collapsed to zero and never came back"


def test_learner_is_a_no_op_when_everything_is_off(panel, equal_weights):
    cfg = RiskLearnerConfig(inverse_vol=False, correlation_scaling=False,
                            portfolio_vol_target=0.0, drawdown_guard=0.0)
    out = RiskLearner(cfg).apply(equal_weights, panel)
    pd.testing.assert_frame_equal(out, equal_weights)


def test_learner_records_diagnostics(panel, equal_weights):
    cfg = RiskLearnerConfig(portfolio_vol_target=0.15, drawdown_guard=0.10)
    learner = RiskLearner(cfg)
    learner.apply(equal_weights, panel)
    assert {"correlation_scale", "vol_scale", "drawdown_scale", "total_scale"} <= set(
        learner.diagnostics_
    )
    assert (learner.diagnostics_["total_scale"] <= cfg.max_scale + 1e-9).all()


def test_learner_output_is_causal(panel, equal_weights):
    """Every risk estimate is trailing and lagged - poisoning the future is inert."""
    from tests.test_no_lookahead import poison_panel

    cfg = RiskLearnerConfig(portfolio_vol_target=0.15, drawdown_guard=0.10)
    clean = RiskLearner(cfg).apply(equal_weights, panel)
    dirty = RiskLearner(cfg).apply(equal_weights, poison_panel(panel, cut=600, seed=11))
    pd.testing.assert_frame_equal(clean.iloc[:600], dirty.iloc[:600], rtol=0, atol=0)


def test_correlation_scale_sits_near_one_for_a_normal_book(panel, equal_weights):
    """A book behaving like its own norm must not be shrunk.

    An earlier version compared the diversification ratio against a fixed
    constant of 2.0 and collapsed a perfectly ordinary equity book from 27% to
    3% volatility. Calibrating against the book's own trailing median makes the
    scale relative by construction.
    """
    scale = correlation_scale(equal_weights, panel.close).tail(300)
    assert 0.7 < scale.median() < 1.4, f"a normal book was rescaled to {scale.median():.2f}"


def test_correlation_scale_still_reacts_to_a_regime_change():
    """It must stay responsive, not just be pinned to 1.0."""
    idx = pd.date_range("2020-01-01", periods=1200, freq="1D", tz="UTC")
    rng = np.random.default_rng(5)
    common = rng.normal(0, 0.012, 1200)
    cols = {}
    for i in range(8):
        idio = rng.normal(0, 0.012, 1200)
        # independent for the first half, then everything moves together
        mixed = np.concatenate([idio[:600], common[600:] + 0.2 * idio[600:]])
        cols[f"S{i}"] = 100 * np.exp(np.cumsum(mixed))
    close = pd.DataFrame(cols, index=idx)
    w = pd.DataFrame(1.0 / 8, index=idx, columns=close.columns)
    scale = correlation_scale(w, close)
    assert scale.iloc[800:].mean() < scale.iloc[400:600].mean(), "no reaction to correlation spiking"
