"""Tests for the reusable evaluation suite."""

from __future__ import annotations

import numpy as np
import pytest
from ml_eval_suite import (
    calibration_curve_data,
    detect_target_leakage,
    evaluate_estimator,
    learning_curve_data,
    permutation_importance_data,
)
from ml_eval_suite.evaluation import cross_validation_metrics
from sklearn.base import clone
from sklearn.compose import make_column_transformer
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


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


def test_importance_is_measured_on_a_holdout_not_fit_rows() -> None:
    x, y = dataset()
    estimator = DecisionTreeClassifier(random_state=8)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=3, stratify=y
    )
    expected = permutation_importance(
        clone(estimator).fit(x_train, y_train),
        x_test,
        y_test,
        n_repeats=10,
        random_state=3,
        scoring="accuracy",
        n_jobs=1,
    )
    result = permutation_importance_data(estimator, x, y, random_state=3)
    np.testing.assert_allclose(result["importances"], expected.importances)
    assert not hasattr(estimator, "tree_")


def test_learning_curve_handles_sorted_binary_labels() -> None:
    x, y = dataset()
    order = np.argsort(y)
    result = learning_curve_data(
        make_pipeline(StandardScaler(), LogisticRegression()), x[order], y[order], cv=3
    )
    assert np.isfinite(result["validation_mean"]).all()


def test_calibration_rejects_multiclass_instead_of_silently_using_column_one() -> None:
    x, _ = dataset()
    with pytest.raises(ValueError, match="binary"):
        calibration_curve_data(LogisticRegression(), x, np.arange(len(x)) % 3)


def test_leakage_threshold_and_nonfinite_inputs_are_validated() -> None:
    x, y = dataset()
    with pytest.raises(ValueError, match="threshold"):
        detect_target_leakage(x, y, x, y, threshold=1.2)
    with pytest.raises(ValueError, match="finite"):
        detect_target_leakage(x * np.nan, y, x, y)


def test_regressor_report_explicitly_skips_probability_calibration() -> None:
    x, _ = dataset()
    result = evaluate_estimator(LinearRegression(), x, x[:, 0] * 2 + x[:, 1])
    assert result["cv"]["scoring"] == "neg_mean_squared_error"
    assert result["calibration"]["status"] == "not_applicable"
    assert np.isfinite(result["learning_curve"]["validation_mean"]).all()


def test_mixed_type_pipeline_reports_unsupported_correlation_check() -> None:
    x, y = dataset()
    mixed = np.empty((len(x), 3), dtype=object)
    mixed[:, :2] = x[:, :2]
    mixed[:, 2] = np.where(x[:, 2] > 0, "a", "b")
    estimator = make_pipeline(
        make_column_transformer(
            (StandardScaler(), [0, 1]), (OneHotEncoder(handle_unknown="ignore"), [2])
        ),
        LogisticRegression(),
    )
    report = evaluate_estimator(estimator, mixed, y)
    assert report["leakage"]["status"] == "unavailable"
    assert np.isfinite(report["cv"]["scores"]).all()


def test_evaluate_estimator_composes_all_outputs() -> None:
    x, y = dataset()
    result = evaluate_estimator(LogisticRegression(max_iter=500), x, y)

    assert {
        "cv",
        "learning_curve",
        "calibration",
        "permutation_importance",
        "leakage",
    } <= result.keys()
