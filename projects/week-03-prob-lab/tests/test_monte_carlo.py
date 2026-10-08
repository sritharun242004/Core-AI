"""Monte Carlo tests. Zero-variance sampling is the Review Focus #1 anchor."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st


def test_pi_converges():
    from prob_lab.monte_carlo import estimate_pi

    est = estimate_pi(n=200_000, seed=42)
    assert abs(est - math.pi) < 0.02


def test_expectation_of_identity_is_mean():
    from prob_lab.monte_carlo import expectation

    rng = np.random.default_rng(7)
    val = expectation(fn=lambda x: x, sampler=lambda size: rng.normal(0, 1, size), n=50_000)
    assert abs(val - 0.0) < 0.02


@given(st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False))
@settings(max_examples=30, deadline=None)
def test_expectation_survives_constant_samplers(const: float) -> None:
    """Review Focus #1 — zero-variance input; must not divide-by-zero or NaN."""
    from prob_lab.monte_carlo import expectation

    val = expectation(
        fn=lambda x: x * 2,
        sampler=lambda size: np.full(size, const, dtype=float),
        n=100,
    )
    assert math.isfinite(val)
    assert abs(val - const * 2) < 1e-10
