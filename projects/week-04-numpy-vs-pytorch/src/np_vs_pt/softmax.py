from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


def softmax_python(x: Sequence[float]) -> list[float]:
    m = max(x)
    exps = [math.exp(xi - m) for xi in x]
    s = sum(exps)
    return [e / s for e in exps]


def softmax_numpy(x: NDArray[np.float64]) -> NDArray[np.float64]:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def softmax_torch(x: Tensor) -> Tensor:
    import torch

    return torch.softmax(x, dim=-1)
