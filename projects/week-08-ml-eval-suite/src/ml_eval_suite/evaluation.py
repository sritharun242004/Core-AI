# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Composable, estimator-agnostic model evaluation helpers.

Each function returns plain dictionaries and NumPy arrays so its output can be
serialized to JSON or passed directly to a plotting notebook.  The suite uses
scikit-learn's own primitives rather than silently implementing a second,
different definition of cross-validation.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.calibration import calibration_curve

from ._typing import (
    CalibrationReport,
    CVReport,
    EvaluationReport,
    Features,
    FitEstimator,
    FloatArray,
    ImportanceReport,
    LeakageReport,
    LearningCurveReport,
    UnavailableReport,
    cross_val_predict,
    cross_val_score,
    learning_curve,
    permutation_importance,
    train_test_split,
)


def cross_validation_metrics(
    estimator: BaseEstimator,
    x: Features,
    y: ArrayLike,
    *,
    cv: int = 5,
    scoring: str = "accuracy",
) -> CVReport:
    """Return each fold's score plus summary statistics."""

    scores = cross_val_score(estimator, x, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scoring": scoring,
        "cv": cv,
    }


def learning_curve_data(
    estimator: BaseEstimator,
    x: Features,
    y: ArrayLike,
    *,
    cv: int = 5,
    scoring: str = "accuracy",
    train_sizes: ArrayLike | None = None,
) -> LearningCurveReport:
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
        shuffle=True,
        random_state=0,
        error_score="raise",
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
    estimator: BaseEstimator,
    x: Features,
    y: ArrayLike,
    *,
    cv: int = 5,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> CalibrationReport:
    """Return reliability-curve points from out-of-fold probabilities.

    Predictions are out-of-fold so the curve does not reward a model for
    calibrating against the rows it just saw during fitting.
    """

    labels = np.asarray(y)
    if labels.ndim != 1 or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("calibration requires binary targets with both labels 0 and 1")
    probabilities = cross_val_predict(estimator, x, y, cv=cv, method="predict_proba", n_jobs=1)[
        :, 1
    ]
    fraction, mean_predicted = calibration_curve(
        y,
        probabilities,
        n_bins=n_bins,
        # sklearn validates the string at runtime; preserve its error contract.
        strategy=cast(Literal["uniform", "quantile"], strategy),
    )
    return {
        "fraction_of_positives": fraction,
        "mean_predicted_value": mean_predicted,
        "n_bins": n_bins,
        "strategy": strategy,
    }


def permutation_importance_data(
    estimator: BaseEstimator,
    x: Features,
    y: ArrayLike,
    *,
    scoring: str = "accuracy",
    n_repeats: int = 10,
    random_state: int = 0,
) -> ImportanceReport:
    """Fit on 75% of rows; measure score decrease on an independent 25%.

    The split is stratified for classifiers. This is a validation holdout,
    not a final test set; importance-driven feature selection still needs a
    separate final assessment. Preprocessing belongs inside the estimator.
    """

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y if is_classifier(estimator) else None,
    )
    fitted = cast(FitEstimator, clone(estimator)).fit(x_train, y_train)
    result = permutation_importance(
        fitted,
        x_valid,
        y_valid,
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


def _safe_correlation(feature: FloatArray, target: FloatArray) -> float:
    if np.std(feature) == 0.0 or np.std(target) == 0.0:
        return 0.0
    value = np.corrcoef(feature, target)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def detect_target_leakage(
    x_train: Features,
    y_train: ArrayLike,
    x_test: Features,
    y_test: ArrayLike,
    *,
    threshold: float = 0.95,
) -> LeakageReport:
    """Flag features almost identical to the target in both data splits.

    This is a deliberately conservative smoke detector, not proof that a
    dataset is clean.  It catches the common accidental ``feature = target``
    mistake before a model is allowed to report a suspiciously good score.
    """

    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    train = np.asarray(x_train, dtype=float)
    test = np.asarray(x_test, dtype=float)
    train_target = np.asarray(y_train, dtype=float)
    test_target = np.asarray(y_test, dtype=float)
    if train_target.ndim != 1 or test_target.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if not all(np.isfinite(a).all() for a in (train, test, train_target, test_target)):
        raise ValueError("leakage checks require finite numeric values")
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
    leaky: list[int] = np.flatnonzero(
        (np.abs(train_correlations) >= threshold) & (np.abs(test_correlations) >= threshold)
    ).tolist()
    return {
        "is_leakage": bool(leaky),
        "leaky_features": leaky,
        "train_correlations": train_correlations,
        "test_correlations": test_correlations,
        "threshold": threshold,
    }


def evaluate_estimator(
    estimator: BaseEstimator,
    x: Features,
    y: ArrayLike,
    *,
    cv: int = 5,
    scoring: str | None = None,
) -> EvaluationReport:
    """Evaluate a sklearn classifier/regressor or complete preprocessing Pipeline.

    Calibration is explicitly not applicable without binary 0/1 probabilities.
    Non-numeric/missing data gets an unavailable correlation report, never an
    assertion that it is leakage-free. Use domain-specific audits as well.
    """

    classifier = is_classifier(estimator)
    if scoring is None:
        scoring = "accuracy" if classifier else "neg_mean_squared_error"
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.25, random_state=0, stratify=y if classifier else None
    )
    leakage: LeakageReport | UnavailableReport
    try:
        leakage = detect_target_leakage(x_train, y_train, x_valid, y_valid)
    except (TypeError, ValueError) as exc:
        leakage = {"status": "unavailable", "reason": str(exc)}
    calibratable = (
        classifier and hasattr(estimator, "predict_proba") and np.array_equal(np.unique(y), [0, 1])
    )
    calibration: CalibrationReport | UnavailableReport = (
        calibration_curve_data(estimator, x, y, cv=cv)
        if calibratable
        else {"status": "not_applicable", "reason": "requires binary 0/1 probabilities"}
    )
    return {
        "leakage": leakage,
        "cv": cross_validation_metrics(estimator, x, y, cv=cv, scoring=scoring),
        "learning_curve": learning_curve_data(estimator, x, y, cv=cv, scoring=scoring),
        "calibration": calibration,
        "permutation_importance": permutation_importance_data(estimator, x, y, scoring=scoring),
    }


__all__ = [
    "calibration_curve_data",
    "cross_validation_metrics",
    "detect_target_leakage",
    "evaluate_estimator",
    "learning_curve_data",
    "permutation_importance_data",
]
