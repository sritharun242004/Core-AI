"""Small, device-aware training helpers for the offline character lab."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import cast

import torch
from torch import Tensor, nn

from .model import CharLSTMAttention


def sequence_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Compute token cross entropy while preserving ``(batch, time)`` semantics."""

    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("logits must be (batch, time, vocab) and targets must be (batch, time)")
    if logits.shape[:2] != targets.shape:
        raise ValueError("logits and targets must agree on batch and time dimensions")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    losses = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


def train_epoch(
    model: CharLSTMAttention,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: str | torch.device = "cpu",
    grad_clip: float | None = 1.0,
) -> float:
    """Train one epoch and return an example-weighted token loss."""

    selected = torch.device(device)
    model.to(selected)
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for source_ids, target_ids in loader:
        source_ids = source_ids.to(selected)
        target_ids = target_ids.to(selected)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(source_ids, target_ids)
        loss = sequence_cross_entropy(logits, target_ids)
        # Tensor.backward's optional arguments are untyped in torch's Python wrapper.
        cast(Callable[[], None], loss.backward)()  # pyright: ignore[reportUnknownMemberType]
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        token_count = target_ids.numel()
        total_loss += float(loss.detach()) * token_count
        total_tokens += token_count
    if total_tokens == 0:
        raise ValueError("loader yielded no tokens")
    return total_loss / total_tokens


@torch.no_grad()
def evaluate(
    model: CharLSTMAttention,
    loader: Iterable[tuple[Tensor, Tensor]],
    *,
    device: str | torch.device = "cpu",
) -> float:
    """Return the mean token loss without changing model parameters."""

    selected = torch.device(device)
    model.to(selected)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for source_ids, target_ids in loader:
        logits, _ = model(source_ids.to(selected), target_ids.to(selected))
        token_count = target_ids.numel()
        total_loss += float(sequence_cross_entropy(logits, target_ids.to(selected))) * token_count
        total_tokens += token_count
    if total_tokens == 0:
        raise ValueError("loader yielded no tokens")
    return total_loss / total_tokens


def fit(
    model: CharLSTMAttention,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int = 1,
    device: str | torch.device = "cpu",
    grad_clip: float | None = 1.0,
) -> dict[str, list[float]]:
    """Run deterministic-friendly training and return serialisable metrics."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    history: dict[str, list[float]] = {"loss": []}
    for _ in range(epochs):
        history["loss"].append(
            train_epoch(model, loader, optimizer, device=device, grad_clip=grad_clip)
        )
    return history


train = fit
