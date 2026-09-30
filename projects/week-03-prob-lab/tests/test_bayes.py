from __future__ import annotations

import pytest


def test_posterior_mean_beta_binomial():
    from prob_lab.bayes import posterior_mean

    assert posterior_mean(2, 2, heads=8, tails=2) == pytest.approx(10 / 14)


def test_posterior_mean_uniform_prior():
    from prob_lab.bayes import posterior_mean

    assert posterior_mean(1, 1, heads=3, tails=1) == pytest.approx(4 / 6)


def test_log_bayes_factor_symmetric():
    from prob_lab.bayes import log_bayes_factor

    a = log_bayes_factor(likelihood_a=0.4, likelihood_b=0.1, prior_a=0.5, prior_b=0.5)
    b = log_bayes_factor(likelihood_a=0.1, likelihood_b=0.4, prior_a=0.5, prior_b=0.5)
    assert a == pytest.approx(-b)
