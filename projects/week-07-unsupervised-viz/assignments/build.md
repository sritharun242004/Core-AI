# Build — compare representations and cluster assumptions (2–3 hours)

Extend the notebook or write a new percent-format notebook that:

1. Runs the five-panel workflow on both `iris` and `digits` with a fixed seed. Save both figures with the dataset name in the filename.
2. Add a small metric table with PCA cumulative explained variance, k-means inertia, GMM average maximum responsibility, and the t-SNE/UMAP configuration. Do not treat a lower t-SNE inertia or a prettier plot as a valid comparison; explain why those quantities are not directly comparable.
3. Compare GMM's hard labels to its soft assignments. Print at least three rows whose largest and second-largest responsibilities are close, and inspect those rows in the PCA panel.
4. Add a `max_samples` experiment for digits. Use the same seed and report how runtime and visual structure change at 300, 900, and the full dataset.
5. Add tests before changing behavior. At minimum, test the new metric-table helper's columns and that a fixed seed gives the same table twice.

**Deliverable:** two saved figures, one metric table, and a paragraph distinguishing an embedding used for visualization from a clustering model used for prediction.
