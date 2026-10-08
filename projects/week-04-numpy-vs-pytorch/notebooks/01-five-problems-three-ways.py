# %% [markdown]
# # Five problems, three ways
# Micro-benchmarks: pure Python vs NumPy vs PyTorch on the same problem.

# %%
import time
from collections.abc import Callable

import numpy as np
import torch
from np_vs_pt import matmul

# %% [markdown]
# ## Matmul at 200 by 200
# %%
rng = np.random.default_rng(0)
A = rng.random((200, 200))
B = rng.random((200, 200))


def timed[T](label: str, fn: Callable[[], T]) -> tuple[str, float, T]:
    t = time.perf_counter()
    r = fn()
    return label, time.perf_counter() - t, r


_, t_py, _ = timed("python", lambda: matmul.matmul_python(A.tolist(), B.tolist()))
_, t_np, _ = timed("numpy", lambda: matmul.matmul_numpy(A, B))
_, t_pt, _ = timed("torch", lambda: matmul.matmul_torch(torch.tensor(A), torch.tensor(B)).numpy())

print(f"python: {t_py * 1000:8.1f} ms")
print(f"numpy:  {t_np * 1000:8.1f} ms")
print(f"torch:  {t_pt * 1000:8.1f} ms")
print(f"speedup numpy vs python: {t_py / t_np:.0f} times")
print(f"speedup torch vs numpy:  {t_np / t_pt:.1f} times")
