import numpy as np
import pandas as pd
import pytest

from ready_mix.preparation import (
    NumericPreparation,
    aggregate_tickets,
    cyclical_pair,
    prepare_table,
)


def frame():
    return pd.DataFrame(
        {
            "fecha": ["2052-01-01", "2052-09-01", "2052-11-01"],
            "split": ["train", "val", "test"],
            "assignment_hour": [7, 7, 7],
            "plant_label": ["Plant A"] * 3,
            "demand_m3": [4.0, 5.0, 6.0],
            "ticket_count": [1, 1, 1],
            "avg_ticket_m3": [4.0, 5.0, 6.0],
            "context": [2.0, np.nan, 100.0],
        }
    )


def test_target_is_sum_and_features_are_means_without_target_sources():
    tickets = pd.DataFrame(
        {
            "fecha": ["2052-01-01"] * 3,
            "split": ["train"] * 3,
            "assignment_hour": [7, 7, 9],
            "plant_label": ["Plant A"] * 3,
            "ticket": ["a", "b", "c"],
            "ticket_volume_m3": [2.0, 3.0, 4.0],
            "context": [10.0, 20.0, 30.0],
        }
    )
    table = aggregate_tickets(tickets)
    assert table.demand_m3.tolist() == [5.0, 4.0]
    assert table.context.tolist() == [15.0, 30.0]
    assert table.ticket_count.tolist() == [2, 1]
    assert table.avg_ticket_m3.tolist() == [2.5, 4.0]
    assert len(table) == 2  # No empty hour is inserted.
    assert "ticket_volume_m3" not in table


def test_training_medians_and_target_source_exclusion():
    table, columns = prepare_table(frame())
    assert columns == ["assignment_hour", "context"]
    assert table.context.tolist() == [2.0, 2.0, 100.0]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda f: f.assign(demand_m3=[0, 5, 6]), "positive demand"),
        (lambda f: f.assign(demand_m3=[np.inf, 5, 6]), "positive demand"),
        (lambda f: f.assign(split=["train", "test", "val"]), "overlap"),
        (lambda f: f.iloc[::-1], "chronologically"),
        (lambda f: f.assign(context=np.nan), "finite"),
        (lambda f: f.assign(context=[2, np.inf, 3]), "finite"),
        (lambda f: pd.concat([f, f.iloc[[-1]]]), "Duplicate"),
    ],
)
def test_invalid_prepared_inputs_fail(mutation, match):
    with pytest.raises(ValueError, match=match):
        prepare_table(mutation(frame()))


def test_numeric_preparation_clips_imputes_scales_and_handles_zero_iqr():
    training = pd.DataFrame({"x": [0.0, 10.0, np.nan, 20.0], "constant": [5.0] * 4})
    prep = NumericPreparation.fit(training, {"x"})
    result = prep.transform(pd.DataFrame({"x": [np.nan, 100.0], "constant": [5.0, 99.0]}))
    assert prep.medians["x"] == 10.0
    assert prep.lower["x"] == pytest.approx(0.3)
    assert prep.upper["x"] == pytest.approx(19.7)
    assert result.x.tolist() == [0.0, 2.0]
    assert result.constant.tolist() == [0.0, 0.0]
    with pytest.raises(ValueError, match="observed"):
        NumericPreparation.fit(pd.DataFrame({"x": [np.nan]}), set())


def test_cyclical_encoding():
    assert cyclical_pair(6, 24) == pytest.approx((1, 0))
    assert cyclical_pair(0, 12) == pytest.approx(cyclical_pair(12, 12))
