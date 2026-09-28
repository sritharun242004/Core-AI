"""Differentiable losses built from the primitive Tensor operations."""

from __future__ import annotations

import numpy as np

from .engine import Tensor


def _reduce(values: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be one of: 'none', 'sum', 'mean'")


def cross_entropy(
    logits: Tensor,
    targets: object,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Multiclass cross-entropy with stable log-sum-exp and three reductions.

    ``targets`` contains integer class IDs. The max used for numerical stability
    is a detached NumPy constant; subtracting a constant changes no derivative,
    while keeping exponentials in a safe range.
    """

    if logits.ndim == 1:
        batch_logits = logits.reshape(1, logits.shape[0])
    elif logits.ndim == 2:
        batch_logits = logits
    else:
        raise ValueError("cross_entropy expects logits shaped (batch, classes)")
    labels = np.asarray(targets, dtype=np.int64).reshape(-1)
    if labels.size != batch_logits.shape[0]:
        raise ValueError("targets must contain one class ID per logit row")
    if np.any(labels < 0) or np.any(labels >= batch_logits.shape[1]):
        raise ValueError("target class is outside the logits range")

    shift = Tensor(np.max(batch_logits.data, axis=1, keepdims=True))
    stable = batch_logits - shift
    log_normalizer = stable.exp().sum(axis=1, keepdims=True).log() + shift
    log_probabilities = batch_logits - log_normalizer
    one_hot = np.zeros_like(batch_logits.data)
    one_hot[np.arange(labels.size), labels] = 1.0
    per_row = -(log_probabilities * Tensor(one_hot)).sum(axis=1)
    return _reduce(per_row, reduction)


def mse_loss(prediction: Tensor, target: object, *, reduction: str = "mean") -> Tensor:
    """Mean-squared error with PyTorch-like ``none``/``sum``/``mean`` semantics."""

    per_element = (prediction - Tensor(target)) ** 2
    return _reduce(per_element, reduction)


__all__ = ["cross_entropy", "mse_loss"]
