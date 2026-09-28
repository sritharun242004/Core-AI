from __future__ import annotations
import numpy as np


def matmul_python(a, b):
    n, m = len(a), len(b[0])
    k = len(b)
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            out[i][j] = sum(a[i][p] * b[p][j] for p in range(k))
    return out


def matmul_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def matmul_torch(a, b):
    import torch  # local import so numpy-only users don't need torch
    return a @ b
