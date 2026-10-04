"""Fit candidates, refit the selected family, and return reproducible results."""

from __future__ import annotations

import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from .modeling import RANDOM_SEED, TEST_MODEL_NAME, cross_validate, metrics, tune_models
from .preparation import prepare_table


def run(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    table, columns = prepare_table(frame)
    train, val, test = [table["split"] == split for split in ["train", "val", "test"]]
    x, y = table[columns], table["demand_m3"]
    candidates = tune_models(x.loc[train], y.loc[train], x.loc[val], y.loc[val])
    best = {}
    for candidate in sorted(candidates, key=lambda row: row["val"]["mae"]):
        family = candidate["model"].split(" alpha=")[0].split(" max_depth=")[0]
        best.setdefault(family, candidate)
    # The original pipeline fixes this family after model comparison. Fail clearly if the
    # replica no longer supports that decision; do not silently change families.
    if next(iter(best)) != TEST_MODEL_NAME:
        raise ValueError("Random Forest no longer has the lowest validation MAE")
    final = clone(best[TEST_MODEL_NAME]["estimator"])
    final.fit(x.loc[train | val], y.loc[train | val])
    predictions = final.predict(x.loc[test])
    test_metrics = metrics(y.loc[test], predictions)
    rows = []
    for family, candidate in best.items():
        rows.append(
            {
                "model_family": family,
                "best_run": candidate["model"],
                "hyperparameters": candidate["estimator"].get_params(),
                "train": candidate["train"],
                "validation": candidate["val"],
                "cv": cross_validate(candidate["estimator"], x.loc[train], y.loc[train]),
                "test": test_metrics if family == TEST_MODEL_NAME else None,
            }
        )
    result = {
        "replication_type": "masked replica",
        "model_scope": "plant-hour",
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "random_seed": RANDOM_SEED,
        "cv_folds": 5,
        "split_counts": table["split"].value_counts().to_dict(),
        "split_dates": {
            split: [
                table.loc[table["split"] == split, "fecha"].min(),
                table.loc[table["split"] == split, "fecha"].max(),
            ]
            for split in ["train", "val", "test"]
        },
        "feature_columns": columns,
        "results_table": rows,
        "selected_model": TEST_MODEL_NAME,
        "final_test_metrics": test_metrics,
    }
    output = table.loc[test, ["fecha", "plant_label", "assignment_hour", "demand_m3"]].copy()
    output["prediction_m3"] = predictions
    return result, output
