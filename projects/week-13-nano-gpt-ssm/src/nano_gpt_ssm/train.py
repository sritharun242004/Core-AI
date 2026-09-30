"""Small offline training helpers shared by the transformer and SSM models."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def language_model_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Cross entropy over ``(batch, time, vocabulary)`` next-token logits."""

    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("logits must be (batch, time, vocab), targets must be (batch, time)")
    if logits.shape[:2] != targets.shape:
        raise ValueError("logits and targets must agree on batch and time")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    losses = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


def training_step(
    model: nn.Module,
    input_ids: Tensor,
    target_ids: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    grad_clip: float | None = 1.0,
) -> float:
    """Run one optimizer update and return the detached scalar loss."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = language_model_loss(logits, target_ids)
    loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach())


def train_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    grad_clip: float | None = 1.0,
) -> float:
    """Train over a loader and return a token-weighted mean loss."""

    total_loss = 0.0
    total_tokens = 0
    for input_ids, target_ids in loader:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = language_model_loss(model(input_ids), target_ids)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        count = target_ids.numel()
        total_loss += float(loss.detach()) * count
        total_tokens += count
    if total_tokens == 0:
        raise ValueError("loader yielded no tokens")
    return total_loss / total_tokens


def fit(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int = 1,
    grad_clip: float | None = 1.0,
) -> dict[str, list[float]]:
    """Run a short local fit and return serialisable loss history."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    history = {"loss": []}
    for _ in range(epochs):
        history["loss"].append(train_epoch(model, loader, optimizer, grad_clip=grad_clip))
    return history
