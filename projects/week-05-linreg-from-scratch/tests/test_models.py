"""Numerical contracts for the Week 5 estimators."""

from __future__ import annotations

import numpy as np
import pytest
from linreg_from_scratch import LinearRegression, LogisticRegression


def test_closed_form_recovers_linear_coefficients() -> None:
    x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    y = 2.5 + x @ np.array([1.25, -0.75])
    model = LinearRegression(solver="closed_form").fit(x, y)

    np.testing.assert_allclose(model.coef_, [1.25, -0.75], atol=1e-6)
    assert model.intercept_ == pytest.approx(2.5, abs=1e-6)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-6)


def test_gradient_descent_reaches_closed_form_in_under_500_steps() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 3))
    y = 0.4 + x @ np.array([1.5, -2.0, 0.75])
    model = LinearRegression(solver="gd", learning_rate=0.08, max_iter=500, tol=1e-12).fit(x, y)

    np.testing.assert_allclose(model.coef_, [1.5, -2.0, 0.75], atol=2e-4)
    assert model.n_iter_ < 500
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_logistic_loss_decreases_monotonically() -> None:
    x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(learning_rate=0.2, max_iter=500).fit(x, y)

    differences = np.diff(model.loss_history_)
    assert np.all(differences <= 1e-10)
    assert model.loss_history_[-1] < model.loss_history_[0]
    assert set(model.predict(x)) <= {0, 1}
    assert np.allclose(model.predict_proba(x).sum(axis=1), 1.0)


def test_l2_regularization_shrinks_coefficients() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(120, 2))
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1]
    unregularized = LinearRegression(solver="closed_form", l2=0.0).fit(x, y)
    regularized = LinearRegression(solver="closed_form", l2=20.0).fit(x, y)

    assert np.linalg.norm(regularized.coef_) < np.linalg.norm(unregularized.coef_)


def test_regularized_solvers_minimize_the_same_mean_loss() -> None:
    rng = np.random.default_rng(31)
    x = rng.normal(size=(100, 3))
    y = 4.0 + x @ np.array([2.0, -1.0, 0.5])
    closed = LinearRegression(l2=0.3).fit(x, y)
    gd = LinearRegression(solver="gd", l2=0.3, tol=1e-14, max_iter=2000).fit(x, y)
    np.testing.assert_allclose(closed.coef_, gd.coef_, atol=1e-5)
    assert closed.intercept_ == pytest.approx(gd.intercept_, abs=1e-5)
    assert closed.loss_history_[-1] == pytest.approx(gd.loss_history_[-1], abs=1e-10)


def test_rank_deficient_design_has_a_least_squares_solution() -> None:
    x = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y = np.array([3.0, 5.0, 7.0])
    np.testing.assert_allclose(LinearRegression().fit(x, y).predict(x), y, atol=1e-10)


@pytest.mark.parametrize("estimator", [LinearRegression, LogisticRegression])
def test_rejects_nonfinite_data_and_multitarget_arrays(estimator) -> None:
    with pytest.raises(ValueError, match="finite"):
        estimator().fit([[np.nan], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="one-dimensional"):
        estimator().fit([[0.0], [1.0]], [[0, 1]])
    fitted = estimator().fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        fitted.predict([[np.inf]])


def test_logistic_regularization_shrinks_and_extreme_logits_are_finite() -> None:
    x = np.linspace(-3, 3, 50)[:, None]
    y = (x[:, 0] > 0).astype(float)
    base = LogisticRegression().fit(x, y)
    regularized = LogisticRegression(l2=1).fit(x, y)
    assert np.linalg.norm(regularized.coef_) < np.linalg.norm(base.coef_)
    assert np.isfinite(base.predict_proba([[-1000.0], [1000.0]])).all()


def test_invalid_shapes_fail_early() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        LinearRegression().fit(np.ones(4), np.ones(4))
    with pytest.raises(ValueError, match="binary"):
        LogisticRegression().fit(np.ones((4, 1)), np.array([0, 1, 2, 1]))
