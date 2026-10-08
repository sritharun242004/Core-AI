# %% [markdown]
# # Bayes & entropy tour
# Coin-bias inference + entropy of common distributions.

# %%
from collections.abc import Callable
from typing import Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.text import Text
from numpy.typing import ArrayLike
from prob_lab import entropy, estimate_pi, posterior_mean

print("π estimate at n=100_000:", estimate_pi(100_000, seed=42))

# %% [markdown]
# ## Posterior mean sweeps
# %%
priors = [(1, 1), (2, 2), (10, 10)]
observations = [(0, 0), (3, 7), (30, 70), (300, 700)]
for a, b in priors:
    row = [posterior_mean(a, b, h, t) for h, t in observations]
    print(f"prior Beta({a},{b}): {row}")


# %% [markdown]
# ## Entropy of the "how many heads?" distribution
# %%
# Matplotlib's **kwargs are untyped; describe only this notebook's call forms.
class FigureFactory(Protocol):
    def __call__(self, *, figsize: tuple[float, float]) -> Figure: ...


probs = np.linspace(0.01, 0.99, 50)
H = [entropy([p, 1 - p]) for p in probs]
cast(FigureFactory, plt.figure)(figsize=(6, 3))
cast(Callable[[ArrayLike, ArrayLike], object], plt.plot)(probs, H)
cast(Callable[[str], Text], plt.xlabel)("P(heads)")
cast(Callable[[str], Text], plt.ylabel)("H(p) (nats)")
cast(Callable[[str], Text], plt.title)("Binary entropy peaks at 0.5")
plt.tight_layout()
cast(Callable[[str], None], plt.savefig)("binary_entropy.png")
plt.close()
