from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


def matmul_python(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    n, m = len(a), len(b[0])
    k = len(b)
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            out[i][j] = sum(a[i][p] * b[p][j] for p in range(k))
    return out


def matmul_numpy(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    return a @ b


def matmul_torch(a: Tensor, b: Tensor) -> Tensor:
    return a @ b
