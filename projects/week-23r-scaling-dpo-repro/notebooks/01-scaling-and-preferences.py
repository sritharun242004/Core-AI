# %% [markdown]
# Three actual small-model runs plus an identifiable synthetic fit and DPO.
# %%
import numpy as np
import torch
from scaling_dpo_repro import fit_power_law, paired_seed_interval, run_three_sizes, train_preference

torch.set_num_threads(1)
rows = run_three_sizes(seed=23)
for row in rows:
    print(row)
sizes = np.array([100, 200, 400, 800])
print("Known-law contract:", fit_power_law(sizes, 1.5 + 4 * sizes**-0.3, irreducible=1.5))
print(
    "Measured toy slope (NOT Chinchilla):",
    fit_power_law([r["parameters"] for r in rows], [r["validation_loss"] for r in rows]),
)
print("Paired seed interval:", paired_seed_interval([1, 2, 3], [1.2, 2.1, 3.3]))
print("One-token DPO:", train_preference(seed=23))
# %% [markdown]
# A negative fitted alpha is permitted: more parameters may overfit this tiny
# task. Never fabricate monotonic losses or use a three-point fit to claim a
# jointly identified parameter/data/compute-optimal law.
