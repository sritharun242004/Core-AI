"""Bayes: Beta-Binomial conjugate posterior + Bayes factors."""

from __future__ import annotations
import math


def posterior_mean(prior_a: float, prior_b: float, heads: int, tails: int) -> float:
    """Beta(a, b) prior + Binomial(heads, tails) likelihood → Beta(a+heads, b+tails)
    posterior, whose mean is (a+heads) / (a+b+heads+tails)."""
    assert prior_a > 0 and prior_b > 0, "beta priors must be positive"
    assert heads >= 0 and tails >= 0, "counts must be non-negative"
    a, b = prior_a + heads, prior_b + tails
    return a / (a + b)


def log_bayes_factor(
    likelihood_a: float, likelihood_b: float,
    prior_a: float, prior_b: float,
) -> float:
    """log P(D|A)/P(D|B) — a log-scale ratio of how well each hypothesis explains D.
    Priors let the caller compute log-posterior-odds.

    Guards against log(0) with a floor at -1e300.
    """
    def _log(x: float) -> float:
        return math.log(x) if x > 0 else -1e300
    return _log(likelihood_a) - _log(likelihood_b) + _log(prior_a) - _log(prior_b)
