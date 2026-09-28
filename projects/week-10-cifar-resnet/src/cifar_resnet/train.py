"""Minimal device-aware training utilities for the Week 10 models."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def choose_device(preference: str = "auto") -> torch.device:
    """Select MPS, CUDA, or CPU without assuming a GPU in tests."""

    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed torch (and CUDA if present) for comparable small experiments."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_metrics(logits: Tensor, labels: Tensor, loss: Tensor) -> tuple[float, int]:
    return float(loss.detach().item()) * labels.shape[0], int(
        (logits.argmax(dim=1) == labels).sum().item()
    )


def train_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: str | torch.device = "auto",
    criterion: nn.Module | None = None,
) -> tuple[float, float]:
    """Run one epoch and return sample-weighted loss and accuracy."""

    selected = choose_device(str(device)) if isinstance(device, str) else device
    criterion = criterion or nn.CrossEntropyLoss()
    model.to(selected)
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images, labels = images.to(selected), labels.to(selected)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_loss, batch_correct = _batch_metrics(logits, labels, loss)
        total_loss += batch_loss
        total_correct += batch_correct
        total_examples += labels.shape[0]
    if total_examples == 0:
        raise ValueError("loader yielded no examples")
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    *,
    device: str | torch.device = "auto",
    criterion: nn.Module | None = None,
) -> tuple[float, float]:
    """Evaluate with gradients disabled and return sample-weighted metrics."""

    selected = choose_device(str(device)) if isinstance(device, str) else device
    criterion = criterion or nn.CrossEntropyLoss()
    model.to(selected)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images, labels = images.to(selected), labels.to(selected)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_loss, batch_correct = _batch_metrics(logits, labels, loss)
        total_loss += batch_loss
        total_correct += batch_correct
        total_examples += labels.shape[0]
    if total_examples == 0:
        raise ValueError("loader yielded no examples")
    return total_loss / total_examples, total_correct / total_examples


def fit(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]] | Iterable[tuple[Tensor, Tensor]],
    *,
    epochs: int = 1,
    learning_rate: float = 0.1,
    weight_decay: float = 5e-4,
    device: str | torch.device = "auto",
) -> dict[str, list[float]]:
    """Train a model with SGD and return serialisable per-epoch metrics."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )
    history = {"train_loss": [], "train_accuracy": []}
    for _ in range(epochs):
        loss, accuracy = train_epoch(model, loader, optimizer, device=device)
        history["train_loss"].append(loss)
        history["train_accuracy"].append(accuracy)
    return history
