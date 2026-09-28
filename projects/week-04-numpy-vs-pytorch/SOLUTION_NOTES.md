# Solution notes — numpy-vs-pytorch

- **Max-shift softmax**: `x - x.max()` subtracts the same constant top and bottom (algebraically identical), but keeps every `e^z` argument ≤ 0. Without it, `exp(1000)` overflows to `inf`.
- **Fused cross-entropy**: `log(softmax(x))` = `x - logsumexp(x)`. Framework `cross_entropy(logits, labels)` does this in one pass; never compose `log ∘ softmax` in prod.
- **k-means one step in NumPy**: uses broadcasting `X[:, None, :] - centers[None, :, :]` to get pairwise diffs, sums along the last axis for squared distances, then `argmin`.
- **Top-k**: `np.argsort(-x)[:k]` is the simplest correct answer; `np.argpartition` is faster for large arrays but the ordering isn't guaranteed within the top-k.
- **PyTorch `topk`** returns `(values, indices)` — we discard values; make sure your test compares indices.
- **Local torch imports**: `matmul.py` etc. do `import torch` inside the function so a user who only wants NumPy doesn't fail at import time. Small overhead per call — for hot loops, hoist the import.
