"""Small mechanism reproductions with explicit identifiability and scale limits."""

import copy
import math

import numpy as np
import torch
from torch import nn


def fit_power_law(sizes, losses, *, irreducible: float = 0.0) -> dict[str, float]:
    """Fit L(N)=E+A*N**(-alpha) with FIXED E using log-linear least squares.

    Three sizes do not identify a joint compute-optimal Chinchilla law. E is a
    declared assumption, not fitted to three points; report sensitivity to E.
    """
    x, y = np.asarray(sizes, dtype=float), np.asarray(losses, dtype=float)
    if (
        x.ndim != 1
        or x.shape != y.shape
        or x.size < 3
        or np.unique(x).size < 3
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not math.isfinite(irreducible)
        or irreducible < 0
        or (x <= 0).any()
        or (y <= irreducible).any()
    ):
        raise ValueError("three distinct positive sizes and losses above fixed E required")
    design = np.column_stack([np.ones(x.size), -np.log(x)])
    target = np.log(y - irreducible)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return {
        "amplitude": float(np.exp(coefficients[0])),
        "alpha": float(coefficients[1]),
        "irreducible": irreducible,
        "rmse_log": float(np.sqrt(np.mean((design @ coefficients - target) ** 2))),
    }


def paired_seed_interval(baseline, candidate, *, seed: int = 0, resamples: int = 1000):
    before, after = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    if (
        before.ndim != 1
        or before.shape != after.shape
        or before.size < 2
        or not np.isfinite(before).all()
        or not np.isfinite(after).all()
        or resamples < 1
    ):
        raise ValueError("at least two paired finite seeds and positive resamples required")
    differences = after - before
    rng = np.random.default_rng(seed)
    means = [rng.choice(differences, size=differences.size).mean() for _ in range(resamples)]
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, *, beta: float = 0.1):
    """Stanford Rafailov et al. 2023; arguments are summed completion log-probs."""
    tensors = (chosen, rejected, reference_chosen, reference_rejected)
    if not chosen.numel() or any(value.shape != chosen.shape for value in tensors):
        raise ValueError("nonempty matching score shapes required")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in tensors):
        raise ValueError("log-probabilities must be finite floating-point scores")
    margin = chosen - rejected - (reference_chosen.detach() - reference_rejected.detach())
    return -nn.functional.logsigmoid(beta * margin).mean()


def train_preference(*, seed: int = 0, steps: int = 30) -> dict:
    """One-token completion policy for three synthetic prompts; not an LLM."""
    if steps < 1:
        raise ValueError("positive steps required")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        policy = nn.Embedding(3, 2)
        reference = copy.deepcopy(policy).eval().requires_grad_(False)
        snapshot = reference.weight.detach().clone()
        prompts = torch.arange(3)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
        with torch.no_grad():
            ref = reference(prompts).log_softmax(-1)
            before = policy(prompts).softmax(-1)[:, 0].mean().item()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            scores = policy(prompts).log_softmax(-1)
            loss = dpo_loss(scores[:, 0], scores[:, 1], ref[:, 0], ref[:, 1])
            loss.backward()
            optimizer.step()
        return {
            "before": before,
            "after": policy(prompts).softmax(-1)[:, 0].mean().item(),
            "reference_unchanged": torch.equal(snapshot, reference.weight),
        }


def run_three_sizes(*, seed: int = 0, steps: int = 20) -> list[dict]:
    """Train three causal fixed-context MLP LMs, NOT three Transformer replicas.

    Token contexts are sampled without replacement; target is their sum mod vocabulary.
    Evaluation uses disjoint context IDs from the seeded permutation. Measured
    loss/counts are real; this task's slope is not a Chinchilla exponent.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        ids = torch.randperm(8**3, generator=generator)[:192]
        contexts = torch.stack([ids // 64, (ids // 8) % 8, ids % 8], dim=-1)
        train_x, valid_x = contexts[:64], contexts[64:]
        train_y, valid_y = train_x.sum(-1) % 8, valid_x.sum(-1) % 8
        rows = []
        for width in (8, 16, 32):
            torch.random.default_generator.manual_seed(seed)
            model = nn.Sequential(
                nn.Embedding(8, width),
                nn.Flatten(),
                nn.Linear(3 * width, width),
                nn.GELU(),
                nn.Linear(width, 8),
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(train_x), train_y)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                valid_loss = nn.functional.cross_entropy(model(valid_x), valid_y).item()
            rows.append(
                {
                    "width": width,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "train_tokens": steps * train_x.numel(),
                    "validation_loss": valid_loss,
                    "seed": seed,
                    "task": "synthetic-sum-mod8",
                }
            )
        return rows


__all__ = [
    "dpo_loss",
    "fit_power_law",
    "paired_seed_interval",
    "run_three_sizes",
    "train_preference",
]
