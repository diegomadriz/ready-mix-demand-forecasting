"""Retrain from a prepared plant-hour CSV; write metrics, predictions, and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .pipeline import run


def write_outputs(results: dict, predictions: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    (output / "replica_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    predictions.to_csv(output / "test_predictions.csv", index=False)
    rows = results["results_table"]
    names = [r["model_family"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names[::-1], [r["validation"]["mae"] for r in rows][::-1], color="#326d91")
    ax.set_xlabel("Validation MAE (m³ per active plant-hour)")
    ax.set_title("Masked replica · chronological model comparison")
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=160)
    plt.close(fig)
    records = [
        {
            "model": r["model_family"],
            "validation_mae": r["validation"]["mae"],
            "validation_rmse": r["validation"]["rmse"],
            "validation_smape": r["validation"]["smape"],
            "cv_mae_mean": r["cv"]["mae_mean"],
            "cv_mae_std": r["cv"]["mae_std"],
            "test_mae": r["test"]["mae"] if r["test"] else None,
        }
        for r in rows
    ]
    pd.DataFrame(records).to_csv(output / "model_comparison.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/plant_hour.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    results, predictions = run(pd.read_csv(args.data))
    write_outputs(results, predictions, args.output)
    print(
        json.dumps(
            {
                "selected_model": results["selected_model"],
                "replica_test": results["final_test_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
