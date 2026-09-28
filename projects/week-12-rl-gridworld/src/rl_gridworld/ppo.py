"""PPO-Clip conceptual demonstration, NOT a PPO training implementation.

This module isolates ratios, fixed advantages, and the pessimistic clipped
surrogate. It has no rollout buffer, critic, GAE, minibatch epochs, or KL
monitoring. REINFORCE in policy_gradient.py is the trainable baseline.
"""

import torch
from torch import Tensor


def clipped_surrogate(
    ratios: Tensor, advantages: Tensor, *, clip_epsilon: float = 0.2
) -> Tensor:
    """Return elementwise ``min(r*A, clip(r, 1-eps, 1+eps)*A)``."""

    if ratios.shape != advantages.shape or ratios.numel() == 0:
        raise ValueError("ratios and advantages must have equal, nonempty shapes")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in (0, 1)")
    if not torch.isfinite(ratios).all() or not torch.isfinite(advantages).all():
        raise ValueError("ratios and advantages must be finite")
    if torch.any(ratios < 0):
        raise ValueError("probability ratios cannot be negative")
    fixed_advantages = advantages.detach()
    clipped = ratios.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    return torch.minimum(ratios * fixed_advantages, clipped * fixed_advantages)


def ppo_clip_loss(
    ratios: Tensor, advantages: Tensor, *, clip_epsilon: float = 0.2
) -> Tensor:
    """Negate the mean surrogate because optimizers minimize a loss."""

    return -clipped_surrogate(ratios, advantages, clip_epsilon=clip_epsilon).mean()


def ppo_clip_demo(
    *,
    old_log_probs: Tensor,
    new_log_probs: Tensor,
    advantages: Tensor,
    clip_epsilon: float = 0.2,
) -> dict[str, Tensor | float]:
    """Inspect frozen-old-policy ratios, objective terms, and loss; no training."""

    if old_log_probs.shape != new_log_probs.shape:
        raise ValueError("old and new log probabilities must have matching shapes")
    ratios = (new_log_probs - old_log_probs.detach()).exp()
    objective = clipped_surrogate(ratios, advantages, clip_epsilon=clip_epsilon)
    return {
        "ratios": ratios,
        "unclipped_objective": ratios * advantages.detach(),
        "objective": objective,
        "loss": -objective.mean(),
        "clip_epsilon": clip_epsilon,
    }


ppo_clip_objective = clipped_surrogate
