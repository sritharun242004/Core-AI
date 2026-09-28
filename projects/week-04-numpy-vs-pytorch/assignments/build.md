# Build (2-3 hrs)

Implement k-means-full (10 iterations) in all three flavors:

1. Pure Python (nested loops)
2. NumPy (broadcasting)
3. PyTorch (with `torch.compile` — measure the JIT compile time separately from steady-state)

Run on the same 2-D dataset (e.g. `sklearn.datasets.make_blobs(n_samples=1000, centers=4, random_state=0)`). Plot the trajectory of each cluster center per iteration in one matplotlib figure per impl. Report wall-clock time.

Acceptance:
- NumPy impl within 2× of Torch on CPU (single-threaded).
- Pure Python 100-1000× slower than NumPy.
- All three converge to the same final centers (up to relabeling).
