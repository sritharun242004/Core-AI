"""Shannon entropy, KL divergence, cross-entropy — all in nats, zero-safe."""

from __future__ import annotations
import numpy as np


def entropy(p: np.ndarray) -> float:
    """Shannon H(p) = -Σ p log p. Zeros contribute 0 by convention (0·log 0 = 0)."""
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return float(-(p[mask] * np.log(p[mask])).sum())


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) = Σ p log(p/q). Undefined when q_i = 0 and p_i > 0 (returns inf).
    When p_i = 0 the contribution is 0 by convention."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    assert p.shape == q.shape, "distributions must share shape"
    mask = p > 0
    if np.any((q[mask] <= 0)):
        return float("inf")
    return float((p[mask] * (np.log(p[mask]) - np.log(q[mask]))).sum())


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """H(p, q) = -Σ p log q = H(p) + KL(p||q)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    if np.any((q[mask] <= 0)):
        return float("inf")
    return float(-(p[mask] * np.log(q[mask])).sum())
