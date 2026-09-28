"""Tests for the reusable evaluation suite."""

from __future__ import annotations

import numpy as np
from ml_eval_suite import (
    calibration_curve_data,
    detect_target_leakage,
    evaluate_estimator,
    learning_curve_data,
    permutation_importance_data,
)
from ml_eval_suite.evaluation import cross_validation_metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def dataset() -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=120,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=21,
    )


def test_cv_matches_sklearn_cross_val_score() -> None:
    x, y = dataset()
    estimator = LogisticRegression(max_iter=500)
    expected = cross_val_score(estimator, x, y, cv=5, scoring="roc_auc")
    result = cross_validation_metrics(estimator, x, y, cv=5, scoring="roc_auc")

    np.testing.assert_allclose(result["scores"], expected)
    assert result["mean"] == np.mean(expected)


def test_learning_curve_and_calibration_have_serializable_shapes() -> None:
    x, y = dataset()
    estimator = LogisticRegression(max_iter=500)
    curve = learning_curve_data(estimator, x, y, cv=3, scoring="roc_auc")
    calibration = calibration_curve_data(estimator, x, y, cv=3, n_bins=6)

    assert len(curve["train_sizes"]) == len(curve["train_mean"])
    assert len(curve["train_mean"]) == len(curve["validation_mean"])
    assert len(calibration["fraction_of_positives"]) <= 6
    assert len(calibration["fraction_of_positives"]) == len(calibration["mean_predicted_value"])


def test_permutation_importance_returns_one_row_per_feature() -> None:
    x, y = dataset()
    result = permutation_importance_data(LogisticRegression(max_iter=500), x, y, random_state=3)

    assert result["importances_mean"].shape == (x.shape[1],)
    assert result["importances_std"].shape == (x.shape[1],)


def test_leakage_detector_flags_target_copied_into_feature() -> None:
    x, y = dataset()
    x_train, x_test = x[:80].copy(), x[80:].copy()
    y_train, y_test = y[:80], y[80:]
    x_train[:, 0] = y_train
    x_test[:, 0] = y_test

    report = detect_target_leakage(x_train, y_train, x_test, y_test)
    assert report["is_leakage"] is True
    assert 0 in report["leaky_features"]


def test_evaluate_estimator_composes_all_outputs() -> None:
    x, y = dataset()
    result = evaluate_estimator(LogisticRegression(max_iter=500), x, y)

    assert {"cv", "learning_curve", "calibration", "permutation_importance"} <= result.keys()
