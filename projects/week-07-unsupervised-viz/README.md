# unsupervised-viz — Week 7 reference project

This lab puts five views of the **same** dataset next to one another:

1. **NumPy PCA** — center the matrix, take an SVD, project onto the leading right-singular vectors, and reconstruct.
2. **NumPy k-means** — Lloyd iterations with k-means++ initialization, fixed-seed reproducibility, and explicit empty-cluster repair.
3. **scikit-learn GMM** — Gaussian-mixture EM with soft assignments.
4. **scikit-learn t-SNE** — a probability-preserving two-dimensional visualization.
5. **UMAP (optional)** — a topological embedding when `umap-learn` is installed.

The default datasets are scikit-learn's built-in **iris** and **digits** fixtures, so the notebook is offline and does not download data. The four required methods always run. If UMAP is not installed, the workflow deliberately keeps a fifth panel labelled **UMAP (missing)** instead of failing or silently pretending that UMAP ran.

## Run the tests

From the repository root, use the existing local environment without changing the workspace lockfile:

```bash
cd projects/week-07-unsupervised-viz
PYTHONPATH=src MPLBACKEND=Agg ../../.venv/bin/python -m pytest
```

The same command can be expressed with `uv`'s isolated dependency injection when a local `.venv` is unavailable:

```bash
uv run --no-project \
  --with 'numpy>=2.1' --with 'matplotlib>=3.9' \
  --with 'scikit-learn>=1.5' --with 'pytest>=8.3' \
  env PYTHONPATH=src MPLBACKEND=Agg python -m pytest
```

`umap-learn` is **not** a default or dev dependency. To opt into the fifth method, install the optional extra in a project-local environment:

```bash
uv sync --extra dev --extra umap
# then: uv run pytest
```

## Public API

```python
from unsupervised_viz import KMeans, PCA, run_comparison

pca = PCA(n_components=2).fit(X)
Z = pca.transform(X)
X_reconstructed = pca.inverse_transform(Z)

clusters = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
print(clusters.labels_, clusters.inertia_)

result = run_comparison(
    dataset="iris",  # also: "digits"
    random_state=42,
    include_umap=True,
    make_figure=True,
)
result.figure.savefig("comparison.png", dpi=160)
print(result.methods, result.missing_methods)
```

`ComparisonOutput.embeddings` contains a two-column array for each available panel. k-means and GMM are clustering algorithms rather than projection algorithms, so their plots reuse the PCA coordinates and color the points by their hard cluster IDs. GMM's full soft assignment matrix is available as `result.soft_assignments`; `soft_assignments[i].sum()` is 1.

## What to look for

- **PCA:** `components_` are rows of the right singular-vector matrix of centered data. `explained_variance_ratio_` is computed from squared singular values divided by `n_samples - 1`.
- **k-means:** `random_state=42` gives byte-for-byte repeatable centers and labels. When a cluster receives no rows, the implementation re-seeds its center to the currently farthest row rather than producing a NaN mean.
- **GMM:** a point can have meaningful membership in more than one Gaussian. Inspect the probability matrix, not only `argmax` labels.
- **t-SNE:** neighborhood visualization is not a global metric-preserving projection. Changing perplexity or seed can change the picture without changing the data.
- **UMAP:** optional means optional in both installation and output. The missing panel is part of the contract and makes the environment state explicit.

## The iris ARI ruling

The handoff's suggested `adjusted_rand_score >= 0.7` is not a good unconditional claim for ordinary all-feature k-means. This project therefore tests and documents a reproducible **petal-length/petal-width iris view**, where the class geometry is the intended teaching fixture:

```python
X_petal = load_iris().data[:, 2:4]
model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_petal)
```

On the local environment this gives ARI ≈ **0.886**. The all-feature view can also exceed 0.7 for some seeds, but a single initialization can fall below it (for example, a ten-seed smoke check ranged from about 0.429 to 0.730 with `n_init=1`). The acceptance test intentionally does not cherry-pick a seed or claim that k-means discovers ground-truth species in every representation. It evaluates the documented petal view and leaves the full-feature result as an experiment.

## Project map

- `src/unsupervised_viz/pca.py` — NumPy SVD PCA and inverse transform.
- `src/unsupervised_viz/kmeans.py` — NumPy k-means++/Lloyd implementation.
- `src/unsupervised_viz/workflow.py` — built-in data loading, sklearn GMM/t-SNE, optional UMAP, and the five-panel figure.
- `tests/test_unsupervised.py` — RED→GREEN acceptance tests for reconstruction, variance, reproducibility, empty clusters, ARI, and comparison output.
- `notebooks/01-unsupervised-comparison.py` — percent-format runnable walkthrough.
