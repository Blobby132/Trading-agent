"""Sample-type labelling: two numbers that look identical and are not."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagent.provenance import Claim, Provenance, Sample, combine, table


def _claim(sample: Sample, sharpe: float, **kw) -> Claim:
    return Claim({"sharpe": sharpe, "final_equity": 1000.0},
                 Provenance(sample=sample, dataset="X", n_configurations=10, **kw))


def test_evidential_ordering_is_explicit():
    assert Sample.UNSEEN_PERIOD.evidential_weight > Sample.UNSEEN_ASSET.evidential_weight
    assert Sample.UNSEEN_ASSET.evidential_weight > Sample.HOLDOUT.evidential_weight
    assert Sample.HOLDOUT.evidential_weight > Sample.OUT_OF_SAMPLE.evidential_weight
    assert Sample.OUT_OF_SAMPLE.evidential_weight > Sample.IN_SAMPLE.evidential_weight


def test_header_states_the_search_size():
    """A result quoted without its configuration count is not interpretable."""
    p = Provenance(Sample.OUT_OF_SAMPLE, "BTC-USD", "2017-07-29", "2026-09-12",
                   n_configurations=1216, seeds=4, n_validation_decisions=3)
    text = p.header()
    assert "OUT-OF-SAMPLE" in text
    assert "1,216 configurations" in text
    assert "4 seed(s)" in text
    assert "3 validation decision(s)" in text


def test_table_puts_the_sample_kind_first_and_strongest_on_top():
    claims = [
        _claim(Sample.IN_SAMPLE, 2.0),
        _claim(Sample.UNSEEN_ASSET, 0.1),
        _claim(Sample.OUT_OF_SAMPLE, 0.9),
    ]
    t = table(claims)
    assert list(t.columns)[0] == "sample"
    assert t.iloc[0]["sample"] == Sample.UNSEEN_ASSET.value
    assert t.iloc[-1]["sample"] == Sample.IN_SAMPLE.value


def test_combine_refuses_to_pool_different_kinds():
    """The specific mistake: averaging an in-sample number with an unseen one."""
    claims = [_claim(Sample.IN_SAMPLE, 2.0), _claim(Sample.UNSEEN_ASSET, 0.1)]
    with pytest.raises(ValueError, match="refusing to pool"):
        combine(claims)


def test_combine_works_within_one_kind():
    claims = [_claim(Sample.UNSEEN_ASSET, 0.1), _claim(Sample.UNSEEN_ASSET, 0.5)]
    assert combine(claims)["sharpe"] == pytest.approx(0.3)


def test_combine_of_nothing_is_empty():
    assert combine([]) == {}


def test_membership_and_costs_travel_with_the_claim():
    """Provenance that lives beside the number cannot be forgotten by a reader."""
    p = Provenance(Sample.OUT_OF_SAMPLE, "us_large_cap",
                   membership="StaticMembership - NOT point-in-time", cost_scenario="high")
    assert "NOT point-in-time" in p.header()
    assert "costs=high" in p.header()
