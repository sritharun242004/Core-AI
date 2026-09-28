# linreg-from-scratch — Week 5 reference project

Implement linear and logistic regression without scikit-learn. The project makes the two optimization stories visible: the linear model has a normal-equation solver and gradient descent, while the logistic model minimizes binary cross-entropy with stable logits.

## Run

```bash
uv sync --extra dev
uv run pytest
```

## Public API

```python
from linreg_from_scratch import LinearRegression, LogisticRegression

linear = LinearRegression(solver="gd", learning_rate=0.05).fit(X, y)
linear.predict(X)

classifier = LogisticRegression(l2=0.1).fit(X, labels)
classifier.predict_proba(X)[:, 1]
```

Both estimators expose `coef_`, `intercept_`, `n_iter_`, and `loss_history_`. The implementation is intentionally NumPy-only; use the week’s notebook to compare closed form and iterative fitting, then compare against scikit-learn in the build assignment.
