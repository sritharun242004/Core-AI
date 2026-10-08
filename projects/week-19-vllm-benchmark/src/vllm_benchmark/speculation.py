"""Exact one-step speculative sampling on a finite vocabulary (no model speedup)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _distributions(target: ArrayLike, draft: ArrayLike) -> tuple[FloatArray, FloatArray]:
    target = np.asarray(target, dtype=np.float64)
    draft = np.asarray(draft, dtype=np.float64)
    if target.ndim != 1 or target.size == 0 or target.shape != draft.shape:
        raise ValueError("target and draft must be nonempty equal-length vectors")
    for probabilities in (target, draft):
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0)
        ):
            raise ValueError("probabilities must be finite, nonnegative, and sum to one")
    # Remove only roundoff from valid unit-mass inputs, not arbitrary unnormalized logits.
    return target / target.sum(), draft / draft.sum()


def acceptance_probability(target: ArrayLike, draft: ArrayLike) -> FloatArray:
    """min(1, p(x)/q(x)); define 1 where q=0 (those proposals never occur)."""
    target, draft = _distributions(target, draft)
    ratio = np.divide(target, draft, out=np.ones_like(target), where=draft > 0)
    return np.minimum(1.0, ratio)


def residual_distribution(target: ArrayLike, draft: ArrayLike) -> FloatArray:
    """Normalized positive part of p-q, NOT p and NOT absolute(p-q)."""
    target, draft = _distributions(target, draft)
    residual = np.maximum(target - draft, 0.0)
    mass = residual.sum()
    # If p==q, rejection has zero probability; returning p keeps the API total.
    return residual / mass if mass > 0 else target.copy()


@dataclass(frozen=True)
class SpeculativeResult:
    token: int
    proposed_token: int
    accepted: bool


def speculative_sample(
    target: ArrayLike, draft: ArrayLike, rng: np.random.Generator
) -> SpeculativeResult:
    target, draft = _distributions(target, draft)
    proposal = int(rng.choice(len(draft), p=draft))
    accepted = bool(rng.random() < min(1.0, target[proposal] / draft[proposal]))
    token = (
        proposal
        if accepted
        else int(rng.choice(len(target), p=residual_distribution(target, draft)))
    )
    return SpeculativeResult(token, proposal, accepted)
