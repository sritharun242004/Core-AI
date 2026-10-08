"""Rank-k SVD reconstruction — the arithmetic behind LoRA and PCA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def svd_reconstruct(matrix: NDArray[np.floating], k: int) -> NDArray[np.floating]:
    """Return the best rank-k approximation of `matrix` in Frobenius norm."""
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return np.zeros_like(matrix, dtype=float)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    k_eff = min(k, s.shape[0])
    return (u[:, :k_eff] * s[:k_eff]) @ vt[:k_eff, :]


def compress_image(path: str | Path, k: int) -> NDArray[np.floating]:
    """Load a grayscale image and return its rank-k SVD reconstruction."""
    img = np.asarray(Image.open(path).convert("L"), dtype=float)
    return svd_reconstruct(img, k=k)
