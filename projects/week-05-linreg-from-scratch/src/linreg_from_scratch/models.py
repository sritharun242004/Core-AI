"""Small, inspectable linear and logistic estimators implemented with NumPy.

The implementation intentionally keeps the intercept in an augmented design
matrix while exposing scikit-learn-shaped attributes.  There is no hidden
solver: the linear model can use the normal equation or gradient descent, and
the logistic model uses stable log-loss plus gradient descent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def _design_matrix(x: Array, fit_intercept: bool) -> Array:
    if x.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if not np.isfinite(x).all():
        raise ValueError("X must contain only finite values")
    return np.column_stack((np.ones(x.shape[0]), x)) if fit_intercept else x.copy()


def _validate_xy(x: ArrayLike, y: ArrayLike) -> tuple[Array, Array]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values")
    if x.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of rows")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature")
    return x, y


@dataclass
class LinearRegression:
    """Ordinary least squares with optional L2 regularization.

    Parameters are learned with the exact normal equation by default.  Set
    ``solver="gd"`` to make the optimization path visible in
    ``loss_history_``.  The intercept is never regularized.
    """

    fit_intercept: bool = True
    l2: float = 0.0
    solver: str = "closed_form"
    learning_rate: float = 0.05
    max_iter: int = 1_000
    tol: float = 1e-10

    def __post_init__(self) -> None:
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative")
        if self.solver not in {"closed_form", "gd"}:
            raise ValueError("solver must be 'closed_form' or 'gd'")
        if self.learning_rate <= 0 or self.max_iter <= 0 or self.tol < 0:
            raise ValueError("learning_rate and max_iter must be positive; tol cannot be negative")

    def fit(self, x: ArrayLike, y: ArrayLike) -> LinearRegression:
        x, y = _validate_xy(x, y)
        design = _design_matrix(x, self.fit_intercept)
        penalty = np.eye(design.shape[1]) * self.l2
        if self.fit_intercept:
            penalty[0, 0] = 0.0

        if self.solver == "closed_form":
            # Minimize mean half-MSE + l2/2 * ||w||². Augmented least
            # squares avoids squaring the condition number and handles rank
            # deficiency. Its normal equation has n*l2 on the diagonal.
            augmented = np.vstack((design, np.sqrt(design.shape[0] * penalty)))
            targets = np.concatenate((y, np.zeros(design.shape[1])))
            self._weights = np.linalg.lstsq(augmented, targets, rcond=None)[0]
            self.loss_history_ = [self._loss(design, y, self._weights)]
            self.n_iter_ = 1
        else:
            weights = np.zeros(design.shape[1], dtype=float)
            history = [self._loss(design, y, weights)]
            for _step in range(1, self.max_iter + 1):
                residual = design @ weights - y
                gradient = design.T @ residual / design.shape[0] + penalty @ weights
                candidate = weights - self.learning_rate * gradient
                candidate_loss = self._loss(design, y, candidate)
                # Backtracking prevents a surprising step size from making
                # the teaching curve go uphill.
                rate = self.learning_rate
                while candidate_loss > history[-1] and rate > 1e-12:
                    rate *= 0.5
                    candidate = weights - rate * gradient
                    candidate_loss = self._loss(design, y, candidate)
                weights = candidate
                history.append(candidate_loss)
                if abs(history[-2] - history[-1]) <= self.tol:
                    break
            self._weights = weights
            self.loss_history_ = history
            self.n_iter_ = len(history) - 1

        self._n_features_in = x.shape[1]
        self.intercept_ = float(self._weights[0]) if self.fit_intercept else 0.0
        self.coef_ = self._weights[1:].copy() if self.fit_intercept else self._weights.copy()
        return self

    def _loss(self, design: Array, y: Array, weights: Array) -> float:
        residual = design @ weights - y
        if self.fit_intercept:
            penalty = self.l2 * np.sum(weights[1:] ** 2) / 2
        else:
            penalty = self.l2 * np.sum(weights**2) / 2
        return float(np.mean(residual**2) / 2 + penalty)

    def predict(self, x: ArrayLike) -> Array:
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._n_features_in:
            raise ValueError(f"X must have shape (n_samples, {self._n_features_in})")
        return _design_matrix(x, self.fit_intercept) @ self._weights

    def _check_fitted(self) -> None:
        if not hasattr(self, "_weights"):
            raise RuntimeError("fit must be called before prediction")


@dataclass
class LogisticRegression:
    """Binary logistic regression trained with stable gradient descent."""

    fit_intercept: bool = True
    l2: float = 0.0
    learning_rate: float = 0.1
    max_iter: int = 1_000
    tol: float = 1e-8

    def __post_init__(self) -> None:
        if self.l2 < 0 or self.learning_rate <= 0 or self.max_iter <= 0 or self.tol < 0:
            raise ValueError("l2 >= 0, learning_rate > 0, max_iter > 0, and tol >= 0 are required")

    @staticmethod
    def _sigmoid(z: Array) -> Array:
        # Splitting the branches avoids exp overflow for large negative logits.
        out = np.empty_like(z, dtype=float)
        positive = z >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        out[~positive] = exp_z / (1.0 + exp_z)
        return out

    def _loss(self, design: Array, y: Array, weights: Array) -> float:
        logits = design @ weights
        data_loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        penalty_weights = weights[1:] if self.fit_intercept else weights
        return float(data_loss + self.l2 * np.sum(penalty_weights**2) / 2)

    def fit(self, x: ArrayLike, y: ArrayLike) -> LogisticRegression:
        x, y = _validate_xy(x, y)
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("y must be binary with labels 0 and 1")
        design = _design_matrix(x, self.fit_intercept)
        weights = np.zeros(design.shape[1], dtype=float)
        history = [self._loss(design, y, weights)]
        for _step in range(1, self.max_iter + 1):
            probabilities = self._sigmoid(design @ weights)
            gradient = design.T @ (probabilities - y) / design.shape[0]
            penalty_weights = weights.copy()
            if self.fit_intercept:
                penalty_weights[0] = 0.0
            gradient += self.l2 * penalty_weights
            rate = self.learning_rate
            candidate = weights - rate * gradient
            candidate_loss = self._loss(design, y, candidate)
            while candidate_loss > history[-1] + 1e-14 and rate > 1e-12:
                rate *= 0.5
                candidate = weights - rate * gradient
                candidate_loss = self._loss(design, y, candidate)
            weights = candidate
            history.append(candidate_loss)
            if abs(history[-2] - history[-1]) <= self.tol:
                break

        self._weights = weights
        self.loss_history_ = history
        self.n_iter_ = len(history) - 1
        self._n_features_in = x.shape[1]
        self.intercept_ = float(weights[0]) if self.fit_intercept else 0.0
        self.coef_ = weights[1:].copy() if self.fit_intercept else weights.copy()
        return self

    def predict_proba(self, x: ArrayLike) -> Array:
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._n_features_in:
            raise ValueError(f"X must have shape (n_samples, {self._n_features_in})")
        positive = self._sigmoid(_design_matrix(x, self.fit_intercept) @ self._weights)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, x: ArrayLike) -> NDArray[np.int_]:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def _check_fitted(self) -> None:
        if not hasattr(self, "_weights"):
            raise RuntimeError("fit must be called before prediction")


__all__ = ["LinearRegression", "LogisticRegression"]
