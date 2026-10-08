"""Narrow call contracts for incomplete sklearn stubs, not estimator replacements.

The pinned stubs leave metadata kwargs unknown and do not specialize the
single-score CV return. These protocols describe only the call forms we use.
"""

from typing import Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn import model_selection
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

FloatArray = NDArray[np.float64]


class CrossValScore(Protocol):
    def __call__(
        self,
        estimator: BaseEstimator,
        x: pd.DataFrame,
        y: ArrayLike,
        *,
        cv: int | BaseCrossValidator,
        scoring: str,
        error_score: str,
    ) -> FloatArray: ...


class PipelineFit(Protocol):
    def __call__(
        self,
        x: pd.DataFrame,
        y: object = None,
        **fit_params: object,
    ) -> Pipeline: ...


class PredictProba(Protocol):
    def __call__(self, x: pd.DataFrame, **params: object) -> FloatArray: ...


cross_val_score = cast(CrossValScore, model_selection.cross_val_score)
