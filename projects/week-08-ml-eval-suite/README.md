# ml-eval-suite — Week 8 reference project

A small, reusable evaluation harness for classical ML experiments. Give it a scikit-learn-compatible classifier plus `(X, y)` and it returns cross-validation metrics, learning-curve data, an out-of-fold calibration curve, and permutation importance. A separate leakage detector catches a feature copied from the target in both train and test splits.

## Run

```bash
uv sync --extra dev
uv run pytest
```

## Public API

```python
from ml_eval_suite import evaluate_estimator
from sklearn.linear_model import LogisticRegression

report = evaluate_estimator(LogisticRegression(max_iter=500), X, y, scoring="roc_auc")
print(report["cv"]["mean"])
```

Every result is a plain dictionary with NumPy arrays so a notebook can plot it or serialize a converted version to JSON. The suite is deliberately classification-focused because calibration and ROC-AUC make the train/validation/test distinctions concrete.
