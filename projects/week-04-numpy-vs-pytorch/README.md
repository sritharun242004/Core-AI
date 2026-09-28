# numpy-vs-pytorch — Week 4 reference project

Five problems (matmul, softmax, cross-entropy, k-means step, top-k) implemented three ways (pure Python, NumPy, PyTorch), with a parity test that keeps every pair numerically identical to atol=1e-6.

## Run

```bash
cd projects/week-04-numpy-vs-pytorch
uv sync --extra dev
uv run pytest -v
```

## What's inside

- `src/np_vs_pt/matmul.py` — 3 impls of matrix-matrix multiply.
- `src/np_vs_pt/softmax.py` — 3 impls; NumPy uses the max-shift trick.
- `src/np_vs_pt/crossentropy.py` — NumPy manual + PyTorch's fused op.
- `src/np_vs_pt/kmeans.py` — one k-means iteration in NumPy & PyTorch.
- `src/np_vs_pt/topk.py` — top-k index selection, 3 ways.
- `notebooks/01-five-problems-three-ways.py` — micro-benchmarks.

The parity test is the acceptance gate: rewrite any impl, the tests will catch a numerical drift.
