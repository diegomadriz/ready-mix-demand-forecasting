"""Chronological plant-hour model comparison."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

RANDOM_SEED = 42
TEST_MODEL_NAME = "Random Forest"


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.divide(
        np.abs(y_true - y_pred),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    return float(np.mean(ratio) * 100)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "smape": smape(y_true, y_pred),
    }


def evaluate_candidate(name: str, estimator, X_train, y_train, X_val, y_val) -> dict:
    start = time.perf_counter()
    estimator.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return {
        "model": name,
        "estimator": estimator,
        "training_time_seconds": elapsed,
        "train": metrics(y_train, estimator.predict(X_train)),
        "val": metrics(y_val, estimator.predict(X_val)),
    }


def cross_validate(estimator, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    splitter = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    for train_idx, valid_idx in splitter.split(X_train):
        model = sklearn.base.clone(estimator)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        fold_scores.append(
            mean_absolute_error(y_train.iloc[valid_idx], model.predict(X_train.iloc[valid_idx]))
        )
    return {
        "mae_mean": float(np.mean(fold_scores)),
        "mae_std": float(np.std(fold_scores, ddof=1)),
        "fold_mae": [float(score) for score in fold_scores],
    }


def tune_models(X_train, y_train, X_val, y_val) -> list[dict]:
    candidates = []
    candidates.append(
        evaluate_candidate(
            "Historical Median Baseline",
            DummyRegressor(strategy="median"),
            X_train,
            y_train,
            X_val,
            y_val,
        )
    )

    for alpha in [0.1, 1.0, 10.0, 50.0]:
        candidates.append(
            evaluate_candidate(
                f"Ridge Regression alpha={alpha}",
                Ridge(alpha=alpha, random_state=RANDOM_SEED),
                X_train,
                y_train,
                X_val,
                y_val,
            )
        )

    for alpha in [0.001, 0.01, 0.1, 1.0]:
        for l1_ratio in [0.2, 0.5, 0.8]:
            candidates.append(
                evaluate_candidate(
                    f"Elastic Net alpha={alpha}, l1_ratio={l1_ratio}",
                    ElasticNet(
                        alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=RANDOM_SEED
                    ),
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                )
            )

    for max_depth in [8, 12, None]:
        for min_samples_leaf in [3, 8]:
            candidates.append(
                evaluate_candidate(
                    f"Random Forest max_depth={max_depth}, min_samples_leaf={min_samples_leaf}",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features="sqrt",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                )
            )
    return candidates
