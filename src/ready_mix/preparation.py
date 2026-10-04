"""Train-derived numeric transforms and active plant-hour aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KEYS = ["fecha", "split", "assignment_hour", "plant_label"]
EXCLUDED = {
    "fecha",
    "split",
    "demand_m3",
    "ticket_count",
    "avg_ticket_m3",
    "plant_label",
    "dominant_product_family",
    "ticket",
    "ticket_volume_m3",
}


def cyclical_pair(position: float, period: float) -> tuple[float, float]:
    angle = 2 * np.pi * (position / period)
    return float(np.sin(angle)), float(np.cos(angle))


@dataclass
class NumericPreparation:
    """Fit median filling, clipping, and median/IQR scaling on training rows only.

    Winsorized columns use the first and ninety-ninth percentiles. Other columns
    use training minima/maxima, matching the original pipeline's continuous-feature loop.
    Existing missingness indicators are retained by the caller.
    """

    medians: pd.Series
    lower: pd.Series
    upper: pd.Series
    centers: pd.Series
    iqrs: pd.Series

    @classmethod
    def fit(cls, training: pd.DataFrame, winsorized: set[str]) -> NumericPreparation:
        medians = training.median()
        if medians.isna().any():
            raise ValueError("Every numeric column needs an observed training value")
        filled = training.fillna(medians)
        lower, upper = filled.min(), filled.max()
        for column in winsorized:
            lower[column], upper[column] = filled[column].quantile([0.01, 0.99])
        bounded = filled.clip(lower=lower, upper=upper, axis=1)
        return cls(
            medians,
            lower,
            upper,
            bounded.median(),
            bounded.quantile(0.75) - bounded.quantile(0.25),
        )

    def transform(self, values: pd.DataFrame) -> pd.DataFrame:
        bounded = values.fillna(self.medians).clip(lower=self.lower, upper=self.upper, axis=1)
        return ((bounded - self.centers) / self.iqrs.replace(0, 1)).round(6)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        c for c in frame.columns if c not in EXCLUDED and pd.api.types.is_numeric_dtype(frame[c])
    ]


def aggregate_tickets(tickets: pd.DataFrame) -> pd.DataFrame:
    """Sum delivered volume and average predictors for observed plant-hours.

    Input consists of enriched, transformed tickets with explicit assignment
    hours and masked plant labels. No empty hours are added to the grid.
    """
    predictors = [c for c in feature_columns(tickets) if c not in KEYS]
    grouped = tickets.groupby(KEYS, sort=False)
    target = (
        grouped["ticket_volume_m3"]
        .agg(demand_m3="sum", ticket_count="size", avg_ticket_m3="mean")
        .reset_index()
    )
    hourly = grouped[predictors].mean().reset_index()
    return (
        target.merge(hourly, on=KEYS, validate="one_to_one")
        .sort_values(["fecha", "assignment_hour", "plant_label"])
        .reset_index(drop=True)
    )


def prepare_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate the prepared replica and order rows by date, hour, and plant."""
    table = frame.copy()
    required = {*KEYS, "demand_m3"}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(table.columns))}")
    dates = pd.to_datetime(table["fecha"], format="%Y-%m-%d", errors="raise")
    if not dates.is_monotonic_increasing:
        raise ValueError("Rows must be chronologically ordered")
    if table.duplicated(["fecha", "assignment_hour", "plant_label"]).any():
        raise ValueError("Duplicate plant-hour")
    if set(table["split"]) != {"train", "val", "test"}:
        raise ValueError("Expected nonempty train, val, and test periods")
    periods = [dates[table["split"] == split] for split in ["train", "val", "test"]]
    if not (periods[0].max() < periods[1].min() and periods[1].max() < periods[2].min()):
        raise ValueError("Split dates overlap or are out of order")
    columns = feature_columns(table)
    if not columns:
        raise ValueError("No numeric predictors")
    if not pd.api.types.is_numeric_dtype(table["demand_m3"]):
        raise ValueError("Demand must be numeric")
    if not np.isfinite(table["demand_m3"]).all() or (table["demand_m3"] <= 0).any():
        raise ValueError("Expected finite positive demand for active hours only")
    medians = table.loc[table["split"] == "train", columns].median()
    table[columns] = table[columns].fillna(medians)
    if not np.isfinite(table[columns].to_numpy()).all():
        raise ValueError("Predictors must be finite after training-median filling")
    table = table.sort_values(["fecha", "assignment_hour", "plant_label"]).reset_index(drop=True)
    return table, columns
