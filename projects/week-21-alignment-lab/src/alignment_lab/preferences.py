"""Real tiny DPO optimization, not a PPO/reward-model implementation.

Each response is ONE categorical token, so its token log-probability is also
its summed completion log-probability. No prompt tokens are scored. The DPO
objective follows Stanford's Rafailov et al. (2023), arXiv:2305.18290.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _training_settings(steps: int, lr: float) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    _positive(lr, "lr")


def dpo_loss(
    chosen: Tensor,
    rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    *,
    beta: float = 0.5,
) -> Tensor:
    """Stable reference-relative logistic loss; reference scores are detached."""
    _positive(beta, "beta")
    values = (chosen, rejected, reference_chosen, reference_rejected)
    if not chosen.numel() or any(value.shape != chosen.shape for value in values):
        raise ValueError("log-probabilities must have equal, nonempty shapes")
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in values):
        raise ValueError("log-probabilities must be finite floating-point tensors")
    margin = chosen - rejected - (reference_chosen.detach() - reference_rejected.detach())
    return -functional.logsigmoid(beta * margin).mean()


class TinyPolicy(nn.Module):
    """Eight context IDs, two residual states, three one-token response actions.

    This is a small conditional policy, NOT a pretrained text language model.
    Initialization preserves the caller's CPU RNG state; all examples use CPU.
    """

    def __init__(self, *, seed: int = 21, contexts: int = 8, width: int = 12):
        super().__init__()
        if contexts < 1 or width < 2:
            raise ValueError("contexts must be positive and width at least two")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(contexts, width)
            self.residual = nn.Linear(width, width)
            self.final_norm = nn.LayerNorm(width)
            self.unembedding = nn.Linear(width, 3, bias=False)

    def hidden_states(self, context_ids: Tensor) -> Tensor:
        first = self.embedding(context_ids)
        second = first + torch.tanh(self.residual(first))
        return torch.stack((first, second), dim=-2)

    def forward(self, context_ids: Tensor) -> Tensor:
        return self.unembedding(self.final_norm(self.hidden_states(context_ids)[..., -1, :]))


@dataclass(frozen=True)
class PreferenceBatch:
    contexts: Tensor
    chosen: Tensor
    rejected: Tensor


def preference_fixture() -> PreferenceBatch:
    """Synthetic preferences; training-set improvement is not generalization."""
    chosen = torch.tensor([0, 1, 2, 0, 2, 1, 0, 1])
    return PreferenceBatch(torch.arange(8), chosen, (chosen + 1) % 3)


def freeze_reference(policy: TinyPolicy) -> TinyPolicy:
    """Independent snapshot, eval mode, no trainable parameters or stale grads."""
    reference = copy.deepcopy(policy).eval()
    reference.requires_grad_(False)
    for parameter in reference.parameters():
        parameter.grad = None
    return reference


@dataclass(frozen=True)
class DPOReport:
    initial_loss: float
    final_loss: float
    initial_margin: float
    final_margin: float
    initial_chosen_probability: float
    final_chosen_probability: float
    final_preference_accuracy: float
    losses: tuple[float, ...]


def train_dpo(
    policy: TinyPolicy,
    reference: TinyPolicy,
    batch: PreferenceBatch,
    *,
    steps: int = 100,
    lr: float = 0.03,
    beta: float = 0.5,
) -> DPOReport:
    """Full-batch Adam on policy only, with cached immutable reference scores.

    Requires a separately frozen/eval reference. Shared underlying storage is
    rejected even if wrapped in distinct Parameter objects. No reference mode,
    buffer, parameter or gradient is mutated by this function.
    """
    _training_settings(steps, lr)
    _positive(beta, "beta")
    reference_storage = {p.untyped_storage().data_ptr() for p in reference.parameters()}
    if any(p.untyped_storage().data_ptr() in reference_storage for p in policy.parameters()):
        raise ValueError("policy and reference must not share parameter storage")
    if reference.training or any(p.requires_grad for p in reference.parameters()):
        raise ValueError("reference must be frozen and in eval mode")
    for values in (batch.contexts, batch.chosen, batch.rejected):
        if values.ndim != 1 or not values.numel() or values.dtype != torch.long:
            raise ValueError("preference IDs must be nonempty int64 vectors")
        if values.shape != batch.contexts.shape:
            raise ValueError("preference vectors must have equal shapes")
    if (batch.contexts < 0).any() or (batch.contexts >= policy.embedding.num_embeddings).any():
        raise ValueError("context IDs out of range")
    if any(((ids < 0) | (ids >= 3)).any() for ids in (batch.chosen, batch.rejected)):
        raise ValueError("response IDs out of range")
    if (batch.chosen == batch.rejected).any():
        raise ValueError("chosen and rejected responses must differ")

    def scores(model: TinyPolicy) -> tuple[Tensor, Tensor]:
        logps = model(batch.contexts).log_softmax(-1)
        return (
            logps.gather(1, batch.chosen[:, None]).squeeze(1),
            logps.gather(1, batch.rejected[:, None]).squeeze(1),
        )

    with torch.no_grad():
        ref_chosen, ref_rejected = scores(reference)

    def objective(chosen: Tensor, rejected: Tensor) -> Tensor:
        return dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=beta)

    def measure() -> tuple[float, float, float, float]:
        with torch.no_grad():
            chosen, rejected = scores(policy)
            return (
                objective(chosen, rejected).item(),
                (chosen - rejected).mean().item(),
                chosen.exp().mean().item(),
                (chosen > rejected).float().mean().item(),
            )

    initial = measure()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(*scores(policy))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    final = measure()
    return DPOReport(
        initial[0], final[0], initial[1], final[1], initial[2], final[2], final[3], tuple(losses)
    )
