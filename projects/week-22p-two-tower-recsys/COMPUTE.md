# Compute — yellow curriculum tier, CPU-first reference

- **Core run:** Python 3.13, PyTorch, NumPy; no network, model weights, credentials or dataset download. A few seconds to tens of seconds on a laptop; timing varies by environment.
- **Memory:** tiny fixture is comfortably below 1 GB beyond framework overhead. Notebook sets one Torch CPU thread. No MPS/CUDA assumption.
- **Budget:** $0 incremental cloud cost. No cloud resources are created, so there is nothing to tear down.
- **Scaling boundary:** exhaustive triples use O(positive events × catalog size) storage; dense graph adjacency uses O((users + items)^2). Do **not** point this implementation at MovieLens 25M unchanged. First introduce training-only sampled negatives, sparse propagation, batched scoring and a measured ANN index.
- **Optional data:** provide an already acquired MovieLens `ratings.csv` locally, respecting GroupLens terms. Parser accepts modern CSV only, remaps IDs and filters positive ratings. Create explicit train/validation/test policies; compare the same eligible catalog. Loading the entire file is not streaming preprocessing.
- **Optional accelerators:** learner-selected experiments only; no cloud run is needed to meet this reference's correctness goals. If you rent a GPU, set a spending limit and delete the instance and attached storage afterward. No real-data timing or production throughput is claimed.
