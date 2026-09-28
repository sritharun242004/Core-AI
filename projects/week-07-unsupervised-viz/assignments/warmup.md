# Warm-up — read the geometry (30 minutes)

1. Load `iris` and print the shape, feature names, and per-feature means. Center the matrix by hand with NumPy and verify that every centered column sums to approximately zero.
2. Fit `PCA(n_components=2)` on the centered/scaled iris matrix. Report `components_.shape`, both explained-variance ratios, and the two-dimensional reconstruction MSE. Explain why the MSE is not expected to be zero after dropping components.
3. Run `KMeans(n_clusters=3, random_state=42, n_init=10)` twice on `iris.data[:, 2:4]`. Check that the centers and labels match exactly. Then change only `random_state` and describe what can change and what must not change (the objective definition and finite outputs).
4. Run `run_comparison("iris", random_state=42, make_figure=True)` without installing UMAP. Find the fifth axis and explain why a missing panel is more informative than silently returning four panels.

**Deliverable:** one short table containing PCA variance ratios, reconstruction MSE, k-means inertia, and the UMAP availability message.
