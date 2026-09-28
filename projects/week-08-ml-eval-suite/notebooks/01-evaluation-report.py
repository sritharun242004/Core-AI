# %% [markdown]
# Week 8 — one evaluation report
#
# The notebook uses a built-in dataset so it is runnable offline. Replace the
# estimator with a Pipeline when your real data contains preprocessing.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from ml_eval_suite import evaluate_estimator

# %%
data = load_breast_cancer()
report = evaluate_estimator(
    LogisticRegression(max_iter=2000), data.data, data.target, scoring="roc_auc"
)
print(report["cv"]["scores"])
print("mean ROC-AUC:", report["cv"]["mean"])

# %%
import matplotlib.pyplot as plt
curve = report["learning_curve"]
plt.plot(curve["train_sizes"], curve["train_mean"], label="train")
plt.plot(curve["train_sizes"], curve["validation_mean"], label="validation")
plt.legend()
plt.xlabel("training rows")
plt.ylabel("ROC-AUC")
plt.show()
