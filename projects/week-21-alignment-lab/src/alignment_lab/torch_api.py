"""Narrow signatures for unannotated PyTorch 2.x entry points used by this lab.

These casts cover individual documented operations, not tensors/modules wholesale.
The runtime calls (including RNG preservation and optimizer semantics) are unchanged.
"""

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import Protocol, cast

import torch
from torch import Tensor


class ForkRNG(Protocol):
    def __call__(self, devices: Iterable[int]) -> AbstractContextManager[object]: ...


class QRResult(Protocol):
    @property
    def Q(self) -> Tensor: ...  # noqa: N802


def _entry(owner: object, name: str) -> object:
    entry: object = getattr(owner, name)
    if not callable(entry):
        raise TypeError(f"PyTorch entry point {name} is not callable")
    return entry


fork_rng = cast(ForkRNG, _entry(torch.random, "fork_rng"))
manual_seed = cast(Callable[[int], torch.Generator], _entry(torch, "manual_seed"))
qr = cast(Callable[[Tensor], QRResult], _entry(torch.linalg, "qr"))


def backward(loss: Tensor) -> None:
    cast(Callable[[], None], _entry(loss, "backward"))()


def optimizer_step(optimizer: torch.optim.Optimizer) -> None:
    cast(Callable[[], object], _entry(optimizer, "step"))()
