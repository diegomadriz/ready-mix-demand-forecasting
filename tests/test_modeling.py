import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from ready_mix import modeling


def test_metrics_include_defined_zero_zero_smape():
    values = modeling.metrics(np.array([0, 2, 4]), np.array([0, 1, 6]))
    assert values["mae"] == 1
    assert values["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert values["smape"] == pytest.approx((0 + 2 / 3 + 0.4) / 3 * 100)


def test_candidate_grid_and_parameters(monkeypatch):
    captured = []

    def capture(name, estimator, *args):
        captured.append((name, estimator))
        return estimator

    monkeypatch.setattr(modeling, "evaluate_candidate", capture)
    candidates = modeling.tune_models(None, None, None, None)
    assert len(candidates) == 23
    forests = [e for _, e in captured if isinstance(e, modeling.RandomForestRegressor)]
    assert {(e.max_depth, e.min_samples_leaf) for e in forests} == {
        (depth, leaf) for depth in [8, 12, None] for leaf in [3, 8]
    }
    assert all(
        e.n_estimators == 300 and e.random_state == 42 and e.max_features == "sqrt" for e in forests
    )
    ridge = [e for _, e in captured if isinstance(e, modeling.Ridge)]
    assert {e.alpha for e in ridge} == {0.1, 1.0, 10.0, 50.0}
    elastic = [e for _, e in captured if isinstance(e, modeling.ElasticNet)]
    assert {(e.alpha, e.l1_ratio) for e in elastic} == {
        (a, r) for a in [0.001, 0.01, 0.1, 1.0] for r in [0.2, 0.5, 0.8]
    }
    assert all(e.max_iter == 20000 and e.random_state == 42 for e in elastic)


def test_cv_uses_expanding_ordered_folds_and_sample_std():
    x = pd.DataFrame({"context": np.arange(12)})
    y = pd.Series(np.arange(12, dtype=float))
    result = modeling.cross_validate(DummyRegressor(strategy="median"), x, y)
    expected = [2.0, 3.0, 4.0, 5.0, 6.0]
    assert result["fold_mae"] == expected
    assert result["mae_mean"] == np.mean(expected)
    assert result["mae_std"] == np.std(expected, ddof=1)
