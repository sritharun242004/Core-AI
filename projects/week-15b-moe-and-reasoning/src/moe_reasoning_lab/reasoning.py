"""Verifiable arithmetic, grouped advantages, and a minimal GRPO-like loss."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn

_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_EXPRESSION = re.compile(rf"({_NUMBER})\s*([+*\-/])\s*({_NUMBER})")
_RESPONSE_NUMBER = re.compile(_NUMBER)


def _question_answer(question: str) -> float | None:
    # Deliberately narrow grammar: only a complete "What is a op b?"
    # question is accepted, never an expression embedded in arbitrary text.
    match = re.fullmatch(
        rf"What is\s+({_NUMBER})\s*([+*\-/])\s*({_NUMBER})\s*\?",
        question.strip(),
    )
    if match is None:
        return None
    left, operator, right = float(match.group(1)), match.group(2), float(match.group(3))
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if right == 0:
        return None
    return left / right


def _final_number(response: str) -> float | None:
    text = str(response).strip()
    # Accept a conventional explicit final answer, while rejecting ambiguous
    # responses with several unrelated numbers.
    explicit = re.search(rf"(?:=|answer\s+is)\s*({_NUMBER})\s*$", text, re.IGNORECASE)
    if explicit is not None:
        return float(explicit.group(1))
    if not re.fullmatch(rf"{_NUMBER}", text):
        return None
    return float(text)


def verify_arithmetic(question: str, response: str | int | float, tolerance: float = 1e-5) -> bool:
    """Return whether the final numeric answer verifies against a safe expression."""

    expected = _question_answer(question)
    actual = _final_number(str(response))
    return expected is not None and actual is not None and abs(expected - actual) <= tolerance


def arithmetic_reward(question: str, response: str | int | float) -> float:
    """Binary RLVR-style reward from the deterministic arithmetic verifier."""

    return 1.0 if verify_arithmetic(question, response) else 0.0


def group_relative_advantages(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """Center and scale rewards independently within each sampled group.

    ``rewards`` has shape ``[prompts, group_size]``. Constant-reward groups
    receive all-zero advantages, avoiding an arbitrary update when the
    verifier cannot distinguish any candidate.
    """

    if rewards.ndim != 2 or rewards.shape[0] < 1 or rewards.shape[1] < 1:
        raise ValueError("rewards must have shape [prompts, group_size] with nonempty dimensions")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not rewards.is_floating_point() or rewards.dtype in (torch.float16, torch.bfloat16):
        rewards = rewards.float()
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite")
    constant_groups = torch.all(rewards == rewards[..., :1], dim=-1, keepdim=True)
    mean = rewards.mean(dim=-1, keepdim=True)
    centered = rewards - mean
    standard_deviation = centered.square().mean(dim=-1, keepdim=True).sqrt()
    scaled = centered / standard_deviation.clamp_min(eps)
    active_groups = (~constant_groups) & (standard_deviation > eps)
    return torch.where(active_groups, scaled, torch.zeros_like(scaled))


def grpo_loss(
    log_probs: Tensor,
    rewards: Tensor,
    old_log_probs: Tensor | None = None,
    reference_log_probs: Tensor | None = None,
    clip_epsilon: float = 0.2,
    kl_weight: float = 0.0,
) -> Tensor:
    """Minimal grouped relative policy objective.

    This is intentionally a teaching implementation: it uses verifier rewards,
    within-prompt relative advantages, an optional PPO-style ratio clip, and an
    optional squared log-probability reference penalty. It is not a claim about
    undocumented production details. ``log_probs`` and ``rewards`` share the
    shape ``[prompts, group_size]``.
    """

    if (
        log_probs.ndim != 2
        or log_probs.shape != rewards.shape
        or log_probs.shape[0] < 1
        or log_probs.shape[1] < 1
    ):
        raise ValueError(
            "log_probs and rewards must have the same nonempty rank-2 shape [prompts, group_size]"
        )
    if not 0 <= clip_epsilon < 1 or not math.isfinite(clip_epsilon):
        raise ValueError("clip_epsilon must be finite and in [0, 1)")
    if not math.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError("kl_weight must be finite and non-negative")
    if kl_weight and reference_log_probs is None:
        raise ValueError("reference_log_probs is required when kl_weight is positive")
    if not torch.isfinite(log_probs).all():
        raise ValueError("log_probs must be finite")
    advantages = group_relative_advantages(rewards).detach()
    old_log_probs = log_probs.detach() if old_log_probs is None else old_log_probs.detach()
    if old_log_probs.shape != log_probs.shape:
        raise ValueError("old_log_probs must match log_probs")
    if not torch.isfinite(old_log_probs).all():
        raise ValueError("old_log_probs must be finite")
    ratio = (log_probs - old_log_probs).exp()
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    loss = -surrogate.mean()
    if reference_log_probs is not None:
        if reference_log_probs.shape != log_probs.shape:
            raise ValueError("reference_log_probs must match log_probs")
        if not torch.isfinite(reference_log_probs).all():
            raise ValueError("reference_log_probs must be finite")
        loss = loss + kl_weight * (log_probs - reference_log_probs.detach()).square().mean()
    return loss


class ArithmeticPolicy(nn.Module):
    """Tiny categorical policy over integer answers for local GRPO experiments."""

    def __init__(self, answer_vocab_size: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        if answer_vocab_size < 2:
            raise ValueError("answer_vocab_size must be at least two")
        self.answer_vocab_size = answer_vocab_size
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, answer_vocab_size),
        )

    def forward(self, prompts: Tensor) -> Tensor:
        if prompts.ndim != 2 or prompts.shape[-1] != 3:
            raise ValueError("prompts must have shape [batch, 3] (left, right, operator id)")
        return self.network(prompts.float())


def sample_grouped_answers(
    policy: ArithmeticPolicy,
    prompts: Tensor,
    group_size: int = 4,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample answer ids and return their selected log probabilities."""

    if group_size < 1:
        raise ValueError("group_size must be positive")
    logits = policy(prompts)
    distribution = torch.softmax(logits, dim=-1)
    grouped_distribution = distribution.unsqueeze(1).expand(-1, group_size, -1)
    actions = torch.multinomial(
        grouped_distribution.reshape(-1, policy.answer_vocab_size),
        num_samples=1,
        generator=generator,
    ).reshape(prompts.shape[0], group_size)
    selected = torch.log_softmax(logits, dim=-1).unsqueeze(1).expand_as(grouped_distribution)
    log_probs = selected.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return actions, log_probs


def grpo_step(
    policy: ArithmeticPolicy,
    optimizer: torch.optim.Optimizer,
    prompts: Tensor,
    actions: Tensor,
    rewards: Tensor,
    old_log_probs: Tensor | None = None,
) -> float:
    """Apply one grouped relative-advantage update to an answer policy."""

    if actions.shape != rewards.shape or actions.ndim != 2:
        raise ValueError("actions and rewards must have shape [prompts, group_size]")
    logits = policy(prompts)
    expanded = torch.log_softmax(logits, dim=-1).unsqueeze(1).expand(-1, actions.shape[1], -1)
    log_probs = expanded.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    loss = grpo_loss(log_probs, rewards, old_log_probs=old_log_probs)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def majority_vote(samples: Iterable[str | int | float]) -> str | int | float:
    """Pick the most frequent sample; ties use the first value observed."""

    values = list(samples)
    if not values:
        raise ValueError("majority vote needs at least one sample")
    return Counter(values).most_common(1)[0][0]


def test_time_majority_vote(
    question: str,
    sample_fn: Callable[[str, random.Random], str | int | float],
    num_samples: int = 8,
    seed: int = 0,
) -> tuple[str | int | float, list[str | int | float]]:
    """Sample a question repeatedly and return the deterministic vote."""

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    rng = random.Random(seed)
    samples = [sample_fn(question, rng) for _ in range(num_samples)]
    return majority_vote(samples), samples
