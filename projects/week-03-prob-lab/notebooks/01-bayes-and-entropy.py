# %% [markdown]
# # Bayes & entropy tour
# Coin-bias inference + entropy of common distributions.

# %%
import numpy as np, matplotlib.pyplot as plt
from prob_lab import estimate_pi, posterior_mean, entropy

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
probs = np.linspace(0.01, 0.99, 50)
H = [entropy([p, 1 - p]) for p in probs]
plt.figure(figsize=(6, 3))
plt.plot(probs, H); plt.xlabel("P(heads)"); plt.ylabel("H(p) (nats)"); plt.title("Binary entropy peaks at 0.5")
plt.tight_layout(); plt.savefig("binary_entropy.png"); plt.close()
