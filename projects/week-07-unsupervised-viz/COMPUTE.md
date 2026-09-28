# COMPUTE — unsupervised-viz

- **Tier:** 🟢 local M-series or any modern CPU.
- **Data:** built-in `iris` (150 × 4) and `digits` (1,797 × 64); no network or credentials.
- **Install:** NumPy, matplotlib, and scikit-learn are the required dependencies. `umap-learn` is optional and is intentionally not in the default or dev dependency sets.
- **Typical time:** tests on an M-series CPU are about 10 seconds, dominated by the t-SNE smoke test. The digits notebook can take tens of seconds depending on the BLAS build.
- **Memory:** under a few hundred MB for the built-in datasets and five-panel figure. Reduce `max_samples` in the notebook if experimenting with a larger replacement dataset.
- **Budget:** $0. No GPU or cloud service is needed.

The primary run deliberately uses the repository's existing `.venv` with `PYTHONPATH=src`, or `uv run --no-project --with ...` dependency injection. Do not run a workspace-wide sync just to try this week; it is not necessary for the lab and should not modify the shared `uv.lock`.

Optional UMAP can be enabled locally with `uv sync --extra umap` from this project (or an equivalent project-local environment). Without it, `run_comparison(..., include_umap=True)` still returns a five-panel figure and labels the fifth panel `UMAP (missing)`.
