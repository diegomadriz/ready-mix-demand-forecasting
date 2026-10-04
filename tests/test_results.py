import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ready_mix.modeling import metrics
from ready_mix.preparation import prepare_table

ROOT = Path(__file__).resolve().parents[1]


def test_saved_errors_recompute_from_held_out_predictions():
    results = json.loads((ROOT / "results/replica_metrics.json").read_text())
    predictions = pd.read_csv(ROOT / "results/test_predictions.csv")
    recomputed = metrics(predictions.demand_m3, predictions.prediction_m3)
    assert recomputed == pytest.approx(results["final_test_metrics"], abs=1e-12)
    table, columns = prepare_table(pd.read_csv(ROOT / "data/plant_hour.csv"))
    assert columns == results["feature_columns"]
    test = table.loc[table["split"] == "test"].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        predictions[["fecha", "plant_label", "assignment_hour"]],
        test[["fecha", "plant_label", "assignment_hour"]],
    )
    np.testing.assert_array_equal(predictions.demand_m3, test.demand_m3)
    assert table["split"].value_counts().to_dict() == results["split_counts"]
    rows = results["results_table"]
    assert [r["model_family"] for r in rows] == [
        "Random Forest",
        "Elastic Net",
        "Ridge Regression",
        "Historical Median Baseline",
    ]
    for row in rows:
        assert row["cv"]["mae_mean"] == pytest.approx(np.mean(row["cv"]["fold_mae"]))
        assert row["cv"]["mae_std"] == pytest.approx(np.std(row["cv"]["fold_mae"], ddof=1))
        if row["model_family"] != "Random Forest":
            assert row["test"] is None
