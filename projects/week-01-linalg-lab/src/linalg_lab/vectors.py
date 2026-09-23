"""Vector operations from first principles."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dot(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Return the dot product of two vectors of equal length."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sum(a * b))


def cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Return cos(angle) in [-1, 1]. Zero vectors return 0 by convention."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def project(a: NDArray[np.floating], onto: NDArray[np.floating]) -> NDArray[np.floating]:
    """Project vector `a` onto vector `onto`."""
    denom = dot(onto, onto)
    if denom == 0.0:
        raise ValueError("cannot project onto zero vector")
    scalar = dot(a, onto) / denom
    return scalar * onto
