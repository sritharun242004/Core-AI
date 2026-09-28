from __future__ import annotations
import math
import numpy as np


def softmax_python(x):
    m = max(x)
    exps = [math.exp(xi - m) for xi in x]
    s = sum(exps)
    return [e / s for e in exps]


def softmax_numpy(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def softmax_torch(x):
    import torch
    return torch.softmax(x, dim=-1)
