# %% [markdown]
# Offline companion to the optional image notebook: synthesize a grayscale matrix.
# No image download is needed to test the rank/reconstruction relationship.
# %%
from itertools import pairwise

import numpy as np
from linalg_lab.svd_compress import svd_reconstruct

rng = np.random.default_rng(1)
image = rng.normal(size=(24, 3)) @ rng.normal(size=(3, 24))
errors = []
for rank in (0, 1, 2, 3):
    approximation = svd_reconstruct(image, rank)
    error = float(np.linalg.norm(image - approximation))
    errors.append(error)
    print({"rank": rank, "frobenius_error": error})
assert all(a >= b for a, b in pairwise(errors))
assert errors[-1] < 1e-10
