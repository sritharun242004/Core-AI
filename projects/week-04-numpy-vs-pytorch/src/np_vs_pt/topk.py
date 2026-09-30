from __future__ import annotations

import numpy as np


def topk_python(x, k):
    return sorted(range(len(x)), key=lambda i: -x[i])[:k]


def topk_numpy(x: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-x)[:k]


def topk_torch(x, k: int):
    import torch

    _, idx = torch.topk(x, k)
    return idx
