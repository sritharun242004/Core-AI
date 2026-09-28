# Build — train an MLP on MNIST-shaped data (2–3 hrs)

Build a small experiment around `make_mnist_shaped` and `train_mlp`:

1. Establish a deterministic baseline with a linear model (`hidden=()`).
2. Add one hidden ReLU layer and plot training loss plus train/test accuracy.
3. Run the same experiment with at least two batch sizes and explain the
   learning-rate interaction caused by mean cross-entropy.
4. Add a held-out confusion matrix using only NumPy (or a plotting library if
   available). Which seven-segment classes are most easily confused?
5. Compare one non-trivial gradient graph with PyTorch's `cross_entropy` when
   torch is installed. If torch is not installed, record that the parity check
   was skipped rather than implying it passed.

Use `seed=7` for the main report so another learner can reproduce the curve.
Keep the synthetic-fixture caveat in the report: success here validates the
engine's plumbing, not a real-MNIST benchmark.
