# %% [markdown]
# Week 5 — regression from scratch
#
# This percent-format notebook runs on CPU. It first fits a noiseless linear
# problem, then follows logistic loss while the classifier learns a boundary.

# %%
import numpy as np
import matplotlib.pyplot as plt
from linreg_from_scratch import LinearRegression, LogisticRegression

# %%
rng = np.random.default_rng(4)
X = rng.normal(size=(100, 1))
y = 1.5 + 2.25 * X[:, 0] + rng.normal(scale=0.15, size=100)
model = LinearRegression(solver="gd", learning_rate=0.1).fit(X, y)
print(model.intercept_, model.coef_, model.n_iter_)

# %%
plt.plot(model.loss_history_)
plt.yscale("log")
plt.xlabel("gradient step")
plt.ylabel("half MSE")
plt.show()

# %%
X_binary = np.linspace(-3, 3, 80)[:, None]
y_binary = (X_binary[:, 0] > 0.2).astype(int)
classifier = LogisticRegression(l2=0.01, learning_rate=0.4).fit(X_binary, y_binary)
plt.plot(classifier.loss_history_)
plt.xlabel("gradient step")
plt.ylabel("binary cross-entropy")
plt.show()
