# linalg-lab — design notes

## Why not `numpy.dot` everywhere?
We reimplement `dot` explicitly because the point of the assignment is to feel the sum. NumPy's `dot` calls BLAS SGEMV internally; the timing exercise in the chapter compares.

## Why SVD via `numpy.linalg.svd`?
`np.linalg.svd` calls LAPACK's `gesdd` — an industrial-strength divide-and-conquer routine. Writing a naive SVD by hand is a 3-week project. The chapter uses SVD as a *tool*; the arithmetic gets built later in W7 (unsupervised methods).

## Numerical gotchas
- Zero-length vector → cosine returns 0 by convention (avoid NaN).
- `np.linalg.svd(A, full_matrices=False)` — never pass `True` for tall matrices unless you want an $m \times m$ $\mathbf{U}$.
