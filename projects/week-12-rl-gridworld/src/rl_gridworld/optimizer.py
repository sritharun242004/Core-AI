"""Explicit optimizer choices and a simple coupled L2 penalty."""

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def make_optimizer(
    parameters: Iterable[nn.Parameter],
    *,
    name: str = "adam",
    learning_rate: float = 0.03,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Construct SGD, Adam, or AdamW; AdamW decouples weight decay."""

    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    choices = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if name not in choices:
        raise ValueError("name must be 'sgd', 'adam', or 'adamw'")
    return choices[name](parameters, lr=learning_rate, weight_decay=weight_decay)


def regularization_penalty(
    parameters: Iterable[Tensor], *, coefficient: float = 0.0
) -> Tensor:
    """Return ``coefficient * sum(theta**2)`` (gradient: ``2*coefficient*theta``).

    Add this to a minimized loss for coupled L2. Do not also pass weight decay
    unless intentionally applying two penalties; AdamW decay is a separate step.
    """

    if coefficient < 0:
        raise ValueError("coefficient cannot be negative")
    terms = [parameter.square().sum() for parameter in parameters]
    if not terms:
        raise ValueError("parameters must not be empty")
    return coefficient * torch.stack(terms).sum()


l2_penalty = regularization_penalty
