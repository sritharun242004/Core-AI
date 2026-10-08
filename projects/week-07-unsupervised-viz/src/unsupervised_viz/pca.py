# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""A small, NumPy-only principal component analysis implementation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class PCA:
    """Principal component analysis computed with an SVD of centered data.

    Parameters
    ----------
    n_components:
        ``None`` keeps every available component.  An integer keeps that many
        components, while a float in ``(0, 1]`` keeps enough components to
        reach that fraction of the total explained variance.

    Notes
    -----
    This class intentionally mirrors the small, useful part of scikit-learn's
    PCA API while keeping the numerical work visible.  The learned mean and
    right singular vectors are enough to transform and reconstruct new rows.
    """

    def __init__(self, n_components: int | float | None = None) -> None:
        if isinstance(n_components, bool):
            raise TypeError("n_components must be an integer, float, or None")
        self.n_components = n_components

    def fit(self, x: ArrayLike) -> PCA:
        x_array = _as_matrix(x)
        n_samples, n_features = x_array.shape
        if n_samples < 2:
            raise ValueError("PCA requires at least two samples")

        self.n_samples_ = n_samples
        self.n_features_in_ = n_features
        self.mean_ = x_array.mean(axis=0)
        centered = x_array - self.mean_
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)

        all_explained = (singular_values**2) / (n_samples - 1)
        total_variance = float(all_explained.sum())
        n_components = _component_count(
            self.n_components,
            min(n_samples, n_features),
            all_explained,
            total_variance,
        )

        self.n_components_ = n_components
        self.components_ = np.asarray(right_vectors[:n_components], dtype=float)
        self.singular_values_ = np.asarray(singular_values[:n_components], dtype=float)
        self.explained_variance_ = np.asarray(all_explained[:n_components], dtype=float)
        if total_variance == 0.0:
            self.explained_variance_ratio_ = np.zeros(n_components, dtype=float)
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        return self

    def transform(self, x: ArrayLike) -> FloatArray:
        self._check_is_fitted()
        x_array = _as_matrix(x)
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x_array.shape[1]}")
        return (x_array - self.mean_) @ self.components_.T

    def fit_transform(self, x: ArrayLike) -> FloatArray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x_reduced: ArrayLike) -> FloatArray:
        self._check_is_fitted()
        reduced = _as_matrix(x_reduced)
        if reduced.shape[1] != self.n_components_:
            raise ValueError(f"expected {self.n_components_} components, got {reduced.shape[1]}")
        return reduced @ self.components_ + self.mean_

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError("fit PCA before calling transform or inverse_transform")


# A descriptive alias is convenient when this class is used beside sklearn.decomposition.PCA.
NumpyPCA = PCA


def _as_matrix(x: ArrayLike) -> FloatArray:
    array = np.asarray(x, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"X must be a two-dimensional array, got ndim={array.ndim}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature")
    if not np.isfinite(array).all():
        raise ValueError("X must contain only finite values")
    return array


def _component_count(
    requested: object,
    available: int,
    explained: FloatArray,
    total_variance: float,
) -> int:
    if requested is None:
        return available
    if isinstance(requested, int):
        if not 1 <= requested <= available:
            raise ValueError(f"n_components must be between 1 and {available}, got {requested}")
        return requested
    if isinstance(requested, float):
        if not 0.0 < requested <= 1.0:
            raise ValueError("a variance target must be in (0, 1]")
        if total_variance == 0.0:
            return available
        cumulative = np.cumsum(explained) / total_variance
        return int(np.searchsorted(cumulative, requested, side="left") + 1)
    raise TypeError("n_components must be an integer, float, or None")
