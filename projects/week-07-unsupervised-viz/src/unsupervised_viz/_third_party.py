"""Precisely scoped boundaries for dataset/optional-extension stub gaps."""

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from sklearn import datasets

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


class DatasetBundle(Protocol):
    data: FloatArray
    target: IntArray


# The default loaders return a Bunch; the pinned stubs incorrectly infer tuples.
load_iris = cast(Callable[[], DatasetBundle], datasets.load_iris)
load_digits = cast(Callable[[], DatasetBundle], datasets.load_digits)


class UMAPReducer(Protocol):
    def fit_transform(self, X: FloatArray) -> FloatArray: ...  # noqa: N803


class UMAPModule(Protocol):
    def UMAP(  # noqa: N802
        self,
        *,
        n_components: int,
        n_neighbors: int,
        min_dist: float,
        metric: str,
        random_state: int,
    ) -> UMAPReducer: ...
