"""Deterministic supervised toy task for inspecting MoE optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from .moe import TinyMoEClassifier


@dataclass
class SupervisedTrainResult:
    """Loss trace and model from a tiny in-memory experiment."""

    model: TinyMoEClassifier
    losses: list[float]


def make_toy_supervised_data(n_samples: int = 32, seed: int = 0) -> tuple[Tensor, Tensor]:
    """Make a linearly separable two-class fixture without a data download."""

    if n_samples < 4:
        raise ValueError("n_samples must be at least four")
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n_samples, 2, generator=generator)
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).long()
    return features, labels


def train_supervised_moe(
    model: TinyMoEClassifier,
    features: Tensor,
    labels: Tensor,
    steps: int = 40,
    learning_rate: float = 0.03,
    aux_weight: float = 0.01,
) -> SupervisedTrainResult:
    """Optimize a fixed batch and return losses for a plumbing smoke test."""

    if steps < 1:
        raise ValueError("steps must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = functional.cross_entropy(logits, labels) + aux_weight * model.aux_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return SupervisedTrainResult(model=model, losses=losses)
