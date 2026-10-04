import json

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from ready_mix import cli, pipeline
from ready_mix.modeling import metrics


def data():
    return pd.DataFrame(
        {
            "fecha": [f"2052-01-{i:02}" for i in range(1, 13)] + ["2052-09-01", "2052-11-01"],
            "split": ["train"] * 12 + ["val", "test"],
            "assignment_hour": [7] * 14,
            "plant_label": ["Plant A"] * 14,
            "demand_m3": list(range(1, 13)) + [8, 10],
            "context": list(range(14)),
        }
    )


def candidates(x, y, xv, yv):
    model = DummyRegressor(strategy="median").fit(x, y)
    scores = metrics(yv, model.predict(xv))
    return [
        {
            "model": "Random Forest max_depth=None, min_samples_leaf=3",
            "estimator": model,
            "train": metrics(y, model.predict(x)),
            "val": scores,
        }
    ]


def test_cli_end_to_end_with_lightweight_estimator(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "tune_models", candidates)
    csv = tmp_path / "input.csv"
    data().to_csv(csv, index=False)
    output = tmp_path / "output"
    cli.main(["--data", str(csv), "--output", str(output)])
    results = json.loads((output / "replica_metrics.json").read_text())
    predictions = pd.read_csv(output / "test_predictions.csv")
    assert results["selected_model"] == "Random Forest"
    assert predictions.prediction_m3.tolist() == [7.0]  # Train + validation median.
    assert results["final_test_metrics"]["mae"] == 3.0
    assert len(results["results_table"][0]["cv"]["fold_mae"]) == 5
    assert (output / "model_comparison.png").stat().st_size > 0
    assert (output / "model_comparison.csv").exists()
    assert json.loads(capsys.readouterr().out)["replica_test"]["mae"] == 3.0


def test_pipeline_refuses_unsupported_fixed_family(monkeypatch):
    def baseline(*args):
        candidate = candidates(*args)[0]
        candidate["model"] = "Historical Median Baseline"
        return [candidate]

    monkeypatch.setattr(pipeline, "tune_models", baseline)
    with pytest.raises(ValueError, match="lowest validation"):
        pipeline.run(data())
