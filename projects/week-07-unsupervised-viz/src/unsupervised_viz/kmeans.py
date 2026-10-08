"""NumPy implementation of Lloyd's k-means algorithm."""

from __future__ import annotations

from typing import TypedDict, Unpack, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class KMeansOptions(TypedDict, total=False):
    max_iter: int
    tol: float
    random_state: int | np.random.Generator | None
    n_init: int
    init: str


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class KMeans:
    """Cluster rows with deterministic, optionally k-means++ initialization.

    Empty clusters are repaired by moving the point currently farthest from
    its assigned center into the empty center.  This keeps every center finite
    and makes the edge case observable rather than silently producing NaNs.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | np.random.Generator | None = None,
        n_init: int = 10,
        init: str = "k-means++",
    ) -> None:
        _positive_integer(n_clusters, "n_clusters")
        _positive_integer(max_iter, "max_iter")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        _positive_integer(n_init, "n_init")
        if init not in {"k-means++", "random"}:
            raise ValueError("init must be 'k-means++' or 'random'")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = float(tol)
        self.random_state = random_state
        self.n_init = n_init
        self.init = init

    def fit(self, x: ArrayLike) -> KMeans:
        x_array = _as_matrix(x)
        self.n_features_in_ = x_array.shape[1]
        self.n_samples_in_ = x_array.shape[0]
        root_rng = np.random.default_rng(self.random_state)

        best: tuple[FloatArray, NDArray[np.int_], float, int] | None = None
        for _ in range(self.n_init):
            seed = int(root_rng.integers(0, np.iinfo(np.int64).max))
            candidate = _single_run(
                x_array,
                self.n_clusters,
                max_iter=self.max_iter,
                tol=self.tol,
                rng=np.random.default_rng(seed),
                init=self.init,
            )
            if best is None or candidate[2] < best[2]:
                best = candidate

        # best cannot be None because n_init >= 1.
        assert best is not None
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, x: ArrayLike) -> NDArray[np.int_]:
        self._check_is_fitted()
        x_array = _as_matrix(x)
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x_array.shape[1]}")
        return _assign(x_array, self.cluster_centers_)[0]

    def fit_predict(self, x: ArrayLike) -> NDArray[np.int_]:
        return self.fit(x).labels_

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("fit KMeans before calling predict")


def kmeans(
    x: ArrayLike,
    n_clusters: int,
    **kwargs: Unpack[KMeansOptions],
) -> KMeans:
    """Convenience constructor returning a fitted :class:`KMeans`."""

    return KMeans(n_clusters=n_clusters, **kwargs).fit(x)


def _single_run(
    x: FloatArray,
    n_clusters: int,
    *,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
    init: str,
) -> tuple[FloatArray, NDArray[np.int_], float, int]:
    centers = _initialize_centers(x, n_clusters, rng, init)
    _iteration = 0
    for _iteration in range(1, max_iter + 1):
        labels, distances = _assign(x, centers)
        new_centers = centers.copy()
        for cluster in range(n_clusters):
            members = x[labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
            else:
                # Re-seed from a hard-to-explain row.  It is preferable to
                # copying an arbitrary first row: this gives the orphaned
                # center a chance to claim a meaningful region next round.
                new_centers[cluster] = x[int(np.argmax(distances))]

        shift = float(np.linalg.norm(new_centers - centers, axis=1).max())
        centers = new_centers
        if shift <= tol:
            break

    labels, distances = _assign(x, centers)
    inertia = float(distances.sum())
    return centers, labels, inertia, _iteration


def _initialize_centers(
    x: FloatArray,
    n_clusters: int,
    rng: np.random.Generator,
    init: str,
) -> FloatArray:
    n_samples = x.shape[0]
    if init == "random":
        indices = rng.integers(0, n_samples, size=n_clusters)
        return x[indices].copy()

    centers = np.empty((n_clusters, x.shape[1]), dtype=float)
    first = int(rng.integers(0, n_samples))
    centers[0] = x[first]
    closest = _squared_distances(x, centers[0:1])[:, 0]
    for cluster in range(1, n_clusters):
        total = float(closest.sum())
        if total <= np.finfo(float).eps:
            # All rows are already represented (often because the input has
            # duplicate rows); sampling with replacement is still finite.
            index = int(rng.integers(0, n_samples))
        else:
            probabilities = closest / total
            # NumPy's scalar choice overload leaves its integer result unknown.
            index = int(cast(np.int64, rng.choice(n_samples, p=probabilities)))
        centers[cluster] = x[index]
        closest = np.minimum(closest, _squared_distances(x, centers[cluster : cluster + 1])[:, 0])
    return centers


def _assign(x: FloatArray, centers: FloatArray) -> tuple[NDArray[np.int_], FloatArray]:
    squared = _squared_distances(x, centers)
    labels = np.argmin(squared, axis=1).astype(int)
    distances = squared[np.arange(len(x)), labels]
    return labels, distances


def _squared_distances(x: FloatArray, centers: FloatArray) -> FloatArray:
    differences = x[:, None, :] - centers[None, :, :]
    return np.einsum("nkd,nkd->nk", differences, differences)


def _as_matrix(x: ArrayLike) -> FloatArray:
    array = np.asarray(x, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"X must be a two-dimensional array, got ndim={array.ndim}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature")
    if not np.isfinite(array).all():
        raise ValueError("X must contain only finite values")
    return array
