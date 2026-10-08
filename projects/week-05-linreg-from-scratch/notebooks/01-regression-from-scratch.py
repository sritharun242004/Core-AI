# %% [markdown]
# Week 5 — regression from scratch
#
# This percent-format notebook runs on CPU. It first fits a noiseless linear
# problem, then follows logistic loss while the classifier learns a boundary.

# %%
from collections.abc import Callable
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from linreg_from_scratch import LinearRegression, LogisticRegression
from matplotlib.text import Text
from numpy.typing import ArrayLike

# %%
rng = np.random.default_rng(4)
X = rng.normal(size=(100, 1))
y = 1.5 + 2.25 * X[:, 0] + rng.normal(scale=0.15, size=100)
model = LinearRegression(solver="gd", learning_rate=0.1).fit(X, y)
print(model.intercept_, model.coef_, model.n_iter_)

# %%
# Narrow the incomplete Matplotlib **kwargs stubs to the call forms used here.
cast(Callable[[ArrayLike], object], plt.plot)(model.loss_history_)
cast(Callable[[str], None], plt.yscale)("log")
cast(Callable[[str], Text], plt.xlabel)("gradient step")
cast(Callable[[str], Text], plt.ylabel)("half MSE")
cast(Callable[[], None], plt.show)()

# %%
X_binary = np.linspace(-3, 3, 80)[:, None]
y_binary = (X_binary[:, 0] > 0.2).astype(int)
classifier = LogisticRegression(l2=0.01, learning_rate=0.4).fit(X_binary, y_binary)
cast(Callable[[ArrayLike], object], plt.plot)(classifier.loss_history_)
cast(Callable[[str], Text], plt.xlabel)("gradient step")
cast(Callable[[str], Text], plt.ylabel)("binary cross-entropy")
cast(Callable[[], None], plt.show)()
