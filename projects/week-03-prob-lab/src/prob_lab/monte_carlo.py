"""Monte Carlo estimators — π by rejection, and generic expectations."""

from __future__ import annotations
from typing import Callable
import numpy as np


def estimate_pi(n: int, seed: int = 0) -> float:
    """Fraction of uniform (x,y) points in the unit square that fall inside the
    quarter unit circle, × 4."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2))
    inside = np.count_nonzero((xy ** 2).sum(axis=1) <= 1.0)
    return 4.0 * inside / n


def expectation(
    fn: Callable[[np.ndarray], np.ndarray | float],
    sampler: Callable[[int], np.ndarray],
    n: int,
    seed: int = 0,
) -> float:
    """E[fn(X)] with X drawn from `sampler(n)`."""
    samples = sampler(n)
    values = np.asarray(fn(samples), dtype=float)
    if values.size == 0:
        return 0.0
    return float(values.mean())
