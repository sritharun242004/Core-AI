from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


def topk_python(x: Sequence[float], k: int) -> list[int]:
    return sorted(range(len(x)), key=lambda i: -x[i])[:k]


def topk_numpy(x: NDArray[np.float64], k: int) -> NDArray[np.intp]:
    return np.argsort(-x)[:k]


def topk_torch(x: Tensor, k: int) -> Tensor:
    import torch

    _, idx = torch.topk(x, k)
    return idx
