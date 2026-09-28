# %% [markdown]
# # Offline Titanic-style XGBoost pipeline
#
# This percent-format notebook uses only a deterministic local fixture. It
# compares a linear baseline with XGBoost using fold-local preprocessing, then
# writes a schema-checked submission artifact.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from xgboost_kaggle import (
    evaluate_models,
    fit_model,
    make_submission,
    make_synthetic_titanic,
)

# %%
train, test = make_synthetic_titanic(n_train=900, n_test=180, seed=42)
print(train.shape, test.shape)
print(train.head(3))
print("missing values:\n", train.isna().sum().sort_values(ascending=False).head(8))
print("survival rate:", train["Survived"].mean())

# %% [markdown]
# ## Cross-validation
#
# `evaluate_models` passes a complete preprocessing + estimator Pipeline to
# `cross_val_score`. The imputer and encoder are fitted only on each fold's
# training rows.

# %%
scores = evaluate_models(train, n_splits=3, seed=42)
summary = pd.DataFrame({name: values for name, values in scores.items()})
print(summary)
print(summary.agg(["mean", "std"]))
print("absolute AUC gap:", summary.xgboost.mean() - summary.baseline.mean())

# %%
fig, ax = plt.subplots(figsize=(6, 3))
ax.boxplot(
    [scores["baseline"], scores["xgboost"]],
    tick_labels=["logistic baseline", "XGBoost"],
)
ax.set_ylabel("ROC-AUC")
ax.set_title("Same folds, fold-local preprocessing, seed=42")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Fit and inspect the submission contract

# %%
model = fit_model(train, model="xgboost", seed=42)
submission = make_submission(model, test)
print(submission.head())
print(submission.columns.tolist(), submission.shape)
assert submission.columns.tolist() == ["PassengerId", "Survived"]
assert submission["PassengerId"].tolist() == test["PassengerId"].tolist()
assert set(submission["Survived"]) <= {0, 1}

# %%
output = Path("artifacts/submission.csv")
make_submission(model, test, output)
print(f"wrote {output}; network/authentication was not used")
