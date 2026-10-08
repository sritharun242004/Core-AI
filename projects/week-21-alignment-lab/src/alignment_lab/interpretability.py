"""Small diagnostic exercises, not causal circuit discovery or full Circuits.

The probe and SAE use separately generated synthetic activations. They do not
explain the DPO policy's behavior; the logit lens alone reads that tiny policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .preferences import training_settings
from .torch_api import backward, fork_rng, manual_seed, optimizer_step, qr


def _matrix(values: Tensor) -> None:
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("activations must be a nonempty [examples, dimensions] matrix")
    if not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("activations must be finite floating-point values")


def logit_lens(residual_states: Tensor, final_norm: nn.Module, unembedding: nn.Module) -> Tensor:
    """Decode each state with the model's FINAL normalization and unembedding.

    Returns logits, not probabilities. Apply softmax on the vocabulary axis.
    Intermediate states need not inhabit the final state's predictive basis.
    """
    if residual_states.ndim < 2 or not torch.isfinite(residual_states).all():
        raise ValueError("residual states must be finite with a final hidden dimension")
    return unembedding(final_norm(residual_states))


def probe_fixture(*, seed: int = 21) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Independent Gaussian draws; label is a known synthetic linear direction."""
    generator = torch.Generator().manual_seed(seed)
    train_x = torch.randn(256, 8, generator=generator)
    test_x = torch.randn(128, 8, generator=generator)
    return train_x, (train_x[:, 0] > 0).float(), test_x, (test_x[:, 0] > 0).float()


class LinearProbe(nn.Module):
    mean: Tensor
    scale: Tensor

    def __init__(self, train_x: Tensor, *, seed: int):
        super().__init__()
        self.register_buffer("mean", train_x.mean(0).detach().clone())
        self.register_buffer("scale", train_x.std(0, unbiased=False).clamp_min(1e-6).detach())
        with fork_rng(devices=[]):
            manual_seed(seed)
            self.linear = nn.Linear(train_x.shape[1], 1)

    def forward(self, values: Tensor) -> Tensor:
        return self.linear((values - self.mean) / self.scale).squeeze(-1)


def _labels(values: Tensor, labels: Tensor) -> None:
    _matrix(values)
    if labels.shape != (len(values),) or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must be a binary vector with one label per example")


def fit_linear_probe(
    train_x: Tensor, train_y: Tensor, *, seed: int = 21, steps: int = 200, lr: float = 0.05
) -> LinearProbe:
    """Fit standardization and logistic probe on training data ONLY."""
    _labels(train_x, train_y)
    training_settings(steps, lr)
    train_x, train_y = train_x.detach(), train_y.detach().to(train_x)
    model = LinearProbe(train_x, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.binary_cross_entropy_with_logits(model(train_x), train_y)
        backward(loss)
        optimizer_step(optimizer)
    return model.eval()


@torch.no_grad()
def probe_accuracy(model: LinearProbe, values: Tensor, labels: Tensor) -> float:
    _labels(values, labels)
    return ((model(values) >= 0) == labels.bool()).float().mean().item()


def sae_fixture(*, seed: int = 21) -> tuple[Tensor, Tensor]:
    """Sparse positive combinations of four known directions, with small noise."""
    generator = torch.Generator().manual_seed(seed)
    dictionary = qr(torch.randn(8, 4, generator=generator)).Q.T

    def sample(count: int) -> Tensor:
        codes = torch.zeros(count, 4)
        active = torch.randint(4, (count, 1), generator=generator)
        codes.scatter_(1, active, 0.5 + torch.rand(count, 1, generator=generator))
        return codes @ dictionary + 0.01 * torch.randn(count, 8, generator=generator)

    return sample(192), sample(96)


class TinySAE(nn.Module):
    """ReLU dictionary learner with unit-norm decoder directions.

    Normalization removes the trivial decoder-up/encoder-down L1 scaling escape.
    Features may still split, duplicate or die: sparsity is not semantic truth.
    """

    center: Tensor

    def __init__(self, width: int, features: int, *, seed: int = 21):
        super().__init__()
        if width < 1 or features < 1:
            raise ValueError("width and features must be positive")
        self.register_buffer("center", torch.zeros(width))
        with fork_rng(devices=[]):
            manual_seed(seed)
            self.encoder = nn.Linear(width, features)
            self.decoder = nn.Parameter(torch.randn(features, width))

    @property
    def dictionary(self) -> Tensor:
        return functional.normalize(self.decoder, dim=1)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        codes = functional.relu(self.encoder(values - self.center))
        return codes @ self.dictionary + self.center, codes


def _penalty(coefficient: float) -> None:
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("l1_coefficient must be finite and nonnegative")


def sae_loss(
    values: Tensor, reconstruction: Tensor, codes: Tensor, *, l1_coefficient: float = 0.01
) -> Tensor:
    """Mean elementwise squared error + lambda * mean per-example L1 sum."""
    _penalty(l1_coefficient)
    _matrix(values)
    _matrix(reconstruction)
    _matrix(codes)
    if reconstruction.shape != values.shape or len(codes) != len(values):
        raise ValueError("reconstruction and code batch shapes must match inputs")
    return functional.mse_loss(reconstruction, values) + l1_coefficient * codes.abs().sum(-1).mean()


@dataclass(frozen=True)
class SAETraining:
    model: TinySAE
    initial_reconstruction: float
    final_reconstruction: float
    losses: tuple[float, ...]


def train_sae(
    train_x: Tensor,
    *,
    features: int = 12,
    seed: int = 21,
    steps: int = 250,
    lr: float = 0.03,
    l1_coefficient: float = 0.01,
) -> SAETraining:
    """Optimize reconstruction+L1 on detached training activations, no test input."""
    _matrix(train_x)
    _penalty(l1_coefficient)
    training_settings(steps, lr)
    train_x = train_x.detach()
    model = TinySAE(train_x.shape[1], features, seed=seed)
    model.center.copy_(train_x.mean(0))
    with torch.no_grad():
        initial = functional.mse_loss(model(train_x)[0], train_x).item()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction, codes = model.forward(train_x)
        loss = sae_loss(train_x, reconstruction, codes, l1_coefficient=l1_coefficient)
        backward(loss)
        optimizer_step(optimizer)
        losses.append(loss.item())
    with torch.no_grad():
        final = functional.mse_loss(model(train_x)[0], train_x).item()
    return SAETraining(model.eval(), initial, final, tuple(losses))


@dataclass(frozen=True)
class ActivationMetrics:
    mean_l0: float
    mean_l1: float
    dead_fraction: float
    firing_rates: tuple[float, ...]


@torch.no_grad()
def activation_metrics(codes: Tensor, *, threshold: float = 1e-6) -> ActivationMetrics:
    """L0/firing use abs(code)>threshold; dead means never active on THIS split."""
    _matrix(codes)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    active = codes.abs() > threshold
    rates = active.float().mean(0)
    return ActivationMetrics(
        active.float().sum(-1).mean().item(),
        codes.abs().sum(-1).mean().item(),
        (rates == 0).float().mean().item(),
        tuple(float(rate.item()) for rate in rates.unbind()),
    )
