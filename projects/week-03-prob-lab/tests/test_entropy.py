from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def test_uniform_entropy_is_log_n():
    from prob_lab.entropy import entropy

    for n in (2, 3, 8):
        p = np.full(n, 1 / n)
        assert entropy(p) == pytest.approx(math.log(n))


def test_delta_entropy_is_zero():
    """log(1) + 0·log(0) = 0. Zero-support case must return 0, not NaN — Review Focus #1."""
    from prob_lab.entropy import entropy

    p = np.array([0.0, 1.0, 0.0, 0.0])
    assert entropy(p) == 0.0


def test_kl_positive():
    from prob_lab.entropy import kl_divergence

    p = np.array([0.7, 0.3])
    q = np.array([0.5, 0.5])
    assert kl_divergence(p, q) > 0.0
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-10)


def test_cross_entropy_ge_entropy():
    """H(p, q) >= H(p), with equality iff p == q."""
    from prob_lab.entropy import cross_entropy, entropy

    p = np.array([0.4, 0.6])
    q = np.array([0.5, 0.5])
    assert cross_entropy(p, q) >= entropy(p) - 1e-10


@given(st.floats(min_value=1e-6, max_value=1 - 1e-6))
@settings(max_examples=50, deadline=None)
def test_binary_kl_bounds(a: float) -> None:
    """Binary KL stays finite and non-negative for interior probabilities."""
    from prob_lab.entropy import kl_divergence

    p = np.array([a, 1 - a])
    q = np.array([0.5, 0.5])
    kl = kl_divergence(p, q)
    assert math.isfinite(kl)
    assert kl >= 0.0
