"""Composable, estimator-agnostic model evaluation helpers.

Each function returns plain dictionaries and NumPy arrays so its output can be
serialized to JSON or passed directly to a plotting notebook.  The suite uses
scikit-learn's own primitives rather than silently implementing a second,
different definition of cross-validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict, cross_val_score, learning_curve


def cross_validation_metrics(
    estimator: Any,
    x: ArrayLike,
    y: ArrayLike,
    *,
    cv: int = 5,
    scoring: str = "accuracy",
) -> dict[str, Any]:
    """Return each fold's score plus summary statistics."""

    scores = cross_val_score(estimator, x, y, cv=cv, scoring=scoring, n_jobs=1)
    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scoring": scoring,
        "cv": cv,
    }


def learning_curve_data(
    estimator: Any,
    x: ArrayLike,
    y: ArrayLike,
    *,
    cv: int = 5,
    scoring: str = "accuracy",
    train_sizes: ArrayLike | None = None,
) -> dict[str, Any]:
    """Compute mean and standard deviation for train/validation curves."""

    if train_sizes is None:
        train_sizes = np.linspace(0.2, 1.0, 5)
    sizes, train_scores, validation_scores = learning_curve(
        estimator,
        x,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )
    return {
        "train_sizes": sizes,
        "train_mean": train_scores.mean(axis=1),
        "train_std": train_scores.std(axis=1),
        "validation_mean": validation_scores.mean(axis=1),
        "validation_std": validation_scores.std(axis=1),
        "scoring": scoring,
    }


def calibration_curve_data(
    estimator: Any,
    x: ArrayLike,
    y: ArrayLike,
    *,
    cv: int = 5,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, Any]:
    """Return reliability-curve points from out-of-fold probabilities.

    Predictions are out-of-fold so the curve does not reward a model for
    calibrating against the rows it just saw during fitting.
    """

    probabilities = cross_val_predict(
        estimator, x, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    fraction, mean_predicted = calibration_curve(
        y, probabilities, n_bins=n_bins, strategy=strategy
    )
    return {
        "fraction_of_positives": fraction,
        "mean_predicted_value": mean_predicted,
        "n_bins": n_bins,
        "strategy": strategy,
    }


def permutation_importance_data(
    estimator: Any,
    x: ArrayLike,
    y: ArrayLike,
    *,
    scoring: str = "accuracy",
    n_repeats: int = 10,
    random_state: int = 0,
) -> dict[str, Any]:
    """Fit once, then measure score decrease after shuffling each feature."""

    fitted = clone(estimator).fit(x, y)
    result = permutation_importance(
        fitted,
        x,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return {
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std,
        "importances": result.importances,
        "scoring": scoring,
    }


def _safe_correlation(feature: np.ndarray, target: np.ndarray) -> float:
    if np.std(feature) == 0.0 or np.std(target) == 0.0:
        return 0.0
    value = np.corrcoef(feature, target)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def detect_target_leakage(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_test: ArrayLike,
    y_test: ArrayLike,
    *,
    threshold: float = 0.95,
) -> dict[str, Any]:
    """Flag features almost identical to the target in both data splits.

    This is a deliberately conservative smoke detector, not proof that a
    dataset is clean.  It catches the common accidental ``feature = target``
    mistake before a model is allowed to report a suspiciously good score.
    """

    train = np.asarray(x_train, dtype=float)
    test = np.asarray(x_test, dtype=float)
    train_target = np.asarray(y_train, dtype=float).reshape(-1)
    test_target = np.asarray(y_test, dtype=float).reshape(-1)
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("X_train and X_test must be two-dimensional")
    if train.shape[1] != test.shape[1]:
        raise ValueError("train and test must have the same number of features")
    if train.shape[0] != train_target.size or test.shape[0] != test_target.size:
        raise ValueError("each feature matrix must align with its target")

    train_correlations = np.array(
        [_safe_correlation(train[:, i], train_target) for i in range(train.shape[1])]
    )
    test_correlations = np.array(
        [_safe_correlation(test[:, i], test_target) for i in range(test.shape[1])]
    )
    leaky = np.flatnonzero(
        (np.abs(train_correlations) >= threshold)
        & (np.abs(test_correlations) >= threshold)
    ).tolist()
    return {
        "is_leakage": bool(leaky),
        "leaky_features": leaky,
        "train_correlations": train_correlations,
        "test_correlations": test_correlations,
        "threshold": threshold,
    }


def evaluate_estimator(
    estimator: Any,
    x: ArrayLike,
    y: ArrayLike,
    *,
    cv: int = 5,
    scoring: str = "accuracy",
) -> dict[str, Any]:
    """Run the complete Week 8 report for a classification estimator."""

    return {
        "cv": cross_validation_metrics(estimator, x, y, cv=cv, scoring=scoring),
        "learning_curve": learning_curve_data(estimator, x, y, cv=cv, scoring=scoring),
        "calibration": calibration_curve_data(estimator, x, y, cv=cv),
        "permutation_importance": permutation_importance_data(
            estimator, x, y, scoring=scoring
        ),
    }


__all__ = [
    "calibration_curve_data",
    "cross_validation_metrics",
    "detect_target_leakage",
    "evaluate_estimator",
    "learning_curve_data",
    "permutation_importance_data",
]
