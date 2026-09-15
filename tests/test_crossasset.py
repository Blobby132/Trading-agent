"""Cross-asset validation: frozen parameters, tiers kept apart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagent.crossasset import (
    TRAINED,
    UNSEEN,
    VALIDATION,
    AssetSpec,
    FrozenConfig,
    apply_frozen,
    cross_asset_table,
    freeze_from_walk_forward,
    verdict,
)
from tradingagent.data import synthetic_ohlcv
from tradingagent.engine import ExecutionConfig
from tradingagent.optimize import WalkForwardConfig, walk_forward


@pytest.fixture(scope="module")
def wf_result():
    data = synthetic_ohlcv(1400, seed=31)
    cfg = WalkForwardConfig(train_bars=400, test_bars=150, n_candidates=8, top_k=2,
                            seed=0, verbose=False)
    return walk_forward(data, ExecutionConfig(initial_capital=100.0), cfg)


def test_freezing_picks_the_modal_choice(wf_result):
    frozen = freeze_from_walk_forward(wf_result, source="synthetic")
    assert frozen.total_folds == len(wf_result.chosen)
    assert 1 <= frozen.chosen_in_folds <= frozen.total_folds
    assert frozen.params  # non-empty


def test_freezing_is_deterministic(wf_result):
    a = freeze_from_walk_forward(wf_result, "s")
    b = freeze_from_walk_forward(wf_result, "s")
    assert a.params == b.params and a.chosen_in_folds == b.chosen_in_folds


def test_freezing_refuses_an_empty_walk_forward(wf_result):
    empty = type(wf_result)(
        equity=wf_result.equity, returns=wf_result.returns, folds=wf_result.folds,
        chosen=[], exec_config=wf_result.exec_config, risk_config=wf_result.risk_config,
        weights=wf_result.weights,
    )
    with pytest.raises(ValueError, match="selected nothing"):
        freeze_from_walk_forward(empty, "s")


def test_applying_a_frozen_config_does_not_search(wf_result):
    """The whole point: the same parameters, whatever the data says."""
    frozen = freeze_from_walk_forward(wf_result, "synthetic")
    other = synthetic_ohlcv(1400, seed=99)
    stats_a = apply_frozen(frozen, other, base_exec=ExecutionConfig(initial_capital=100.0))
    stats_b = apply_frozen(frozen, other, base_exec=ExecutionConfig(initial_capital=100.0))
    assert stats_a["final_equity"] == stats_b["final_equity"]
    assert frozen.params == freeze_from_walk_forward(wf_result, "synthetic").params


def test_frozen_config_records_its_provenance(wf_result):
    frozen = freeze_from_walk_forward(wf_result, source="BTC-USD 2015-2026")
    assert "BTC-USD" in frozen.describe()
    assert "folds" in frozen.describe()


def test_cross_asset_table_keeps_tiers_separate(wf_result):
    frozen = freeze_from_walk_forward(wf_result, "synthetic")
    datasets = {
        "TRAIN": synthetic_ohlcv(900, seed=31),
        "VALID": synthetic_ohlcv(900, seed=41),
        "UNSEEN1": synthetic_ohlcv(900, seed=51),
        "UNSEEN2": synthetic_ohlcv(900, seed=61),
    }
    specs = {
        "TRAIN": AssetSpec("TRAIN", TRAINED),
        "VALID": AssetSpec("VALID", VALIDATION),
        "UNSEEN1": AssetSpec("UNSEEN1", UNSEEN),
        "UNSEEN2": AssetSpec("UNSEEN2", UNSEEN),
    }
    table = cross_asset_table(frozen, datasets, specs)
    assert len(table) == 4
    assert set(table["tier"]) == {TRAINED, VALIDATION, UNSEEN}
    # the trained row must be first so a reader cannot mistake it for evidence
    assert table.iloc[0]["tier"] == TRAINED

    v = verdict(table)
    assert set(v) == {TRAINED, VALIDATION, UNSEEN}
    assert v[UNSEEN]["assets"] == 2
    # tiers are summarised independently - no pooled average anywhere
    assert v[TRAINED]["assets"] == 1


def test_cross_asset_table_compares_against_each_asset_own_baseline(wf_result):
    """A rising market lifts any long rule, so the comparison must be per asset."""
    frozen = freeze_from_walk_forward(wf_result, "synthetic")
    datasets = {"A": synthetic_ohlcv(900, seed=71), "B": synthetic_ohlcv(900, seed=81)}
    specs = {"A": AssetSpec("A", UNSEEN), "B": AssetSpec("B", UNSEEN)}
    table = cross_asset_table(frozen, datasets, specs)
    assert "bl_buy_and_hold" in table.columns
    assert "beats_buy_hold" in table.columns
    # each row's baseline must come from its own series, not a shared one
    assert table["bl_buy_and_hold"].nunique() == 2


def test_verdict_reports_the_worst_case_not_just_the_median(wf_result):
    frozen = freeze_from_walk_forward(wf_result, "synthetic")
    datasets = {f"U{i}": synthetic_ohlcv(900, seed=100 + i) for i in range(4)}
    specs = {k: AssetSpec(k, UNSEEN) for k in datasets}
    v = verdict(cross_asset_table(frozen, datasets, specs))
    assert "worst_sharpe" in v[UNSEEN] and "worst_drawdown" in v[UNSEEN]
    assert v[UNSEEN]["worst_sharpe"] <= v[UNSEEN]["median_sharpe"]
