# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Report schemas and narrow contracts for sklearn's incomplete public stubs.

In particular learning_curve returns THREE arrays when return_times=False, and
permutation_importance returns a single Bunch for a single scoring string.
These casts specialize those documented modes without replacing estimators.
"""

from typing import Literal, Protocol, TypedDict, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn import inspection, model_selection
from sklearn.base import BaseEstimator

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
type Features = ArrayLike | pd.DataFrame


class CVReport(TypedDict):
    scores: FloatArray
    mean: float
    std: float
    scoring: str
    cv: int


class LearningCurveReport(TypedDict):
    train_sizes: IntArray
    train_mean: FloatArray
    train_std: FloatArray
    validation_mean: FloatArray
    validation_std: FloatArray
    scoring: str


class CalibrationReport(TypedDict):
    fraction_of_positives: FloatArray
    mean_predicted_value: FloatArray
    n_bins: int
    strategy: str


class ImportanceReport(TypedDict):
    importances_mean: FloatArray
    importances_std: FloatArray
    importances: FloatArray
    scoring: str


class LeakageReport(TypedDict):
    is_leakage: bool
    leaky_features: list[int]
    train_correlations: FloatArray
    test_correlations: FloatArray
    threshold: float


class UnavailableReport(TypedDict):
    status: Literal["unavailable", "not_applicable"]
    reason: str


class EvaluationReport(TypedDict):
    leakage: LeakageReport | UnavailableReport
    cv: CVReport
    learning_curve: LearningCurveReport
    calibration: CalibrationReport | UnavailableReport
    permutation_importance: ImportanceReport


class CrossValScore(Protocol):
    def __call__(
        self,
        estimator: BaseEstimator,
        X: Features,  # noqa: N803
        y: ArrayLike,
        *,
        cv: int,
        scoring: str,
        n_jobs: int = 1,
        error_score: str = "raise",
    ) -> FloatArray: ...


class CrossValPredict(Protocol):
    def __call__(
        self,
        estimator: BaseEstimator,
        X: Features,  # noqa: N803
        y: ArrayLike,
        *,
        cv: int,
        method: Literal["predict_proba"],
        n_jobs: int,
    ) -> FloatArray: ...


class LearningCurve(Protocol):
    def __call__(
        self,
        estimator: BaseEstimator,
        X: Features,  # noqa: N803
        y: ArrayLike,
        *,
        train_sizes: ArrayLike,
        cv: int,
        scoring: str,
        n_jobs: int,
        shuffle: bool,
        random_state: int,
        error_score: str,
    ) -> tuple[IntArray, FloatArray, FloatArray]: ...


class TrainTestSplit(Protocol):
    def __call__(
        self,
        X: Features,  # noqa: N803
        y: ArrayLike,
        *,
        test_size: float,
        random_state: int,
        stratify: ArrayLike | None,
    ) -> tuple[Features, Features, ArrayLike, ArrayLike]: ...


class ImportanceResult(Protocol):
    importances_mean: FloatArray
    importances_std: FloatArray
    importances: FloatArray


class PermutationImportance(Protocol):
    def __call__(
        self,
        estimator: BaseEstimator,
        X: Features,  # noqa: N803
        y: ArrayLike,
        *,
        scoring: str,
        n_repeats: int,
        random_state: int,
        n_jobs: int,
    ) -> ImportanceResult: ...


class FitEstimator(Protocol):
    def fit(self, X: Features, y: ArrayLike) -> BaseEstimator: ...  # noqa: N803


cross_val_score = cast(CrossValScore, model_selection.cross_val_score)
cross_val_predict = cast(CrossValPredict, model_selection.cross_val_predict)
learning_curve = cast(LearningCurve, model_selection.learning_curve)
train_test_split = cast(TrainTestSplit, model_selection.train_test_split)
permutation_importance = cast(PermutationImportance, inspection.permutation_importance)
