# Build (2-3 hrs)

Train a 2-layer MLP on the classic moons dataset from `sklearn.datasets.make_moons(n_samples=100, noise=0.1)`. No sklearn model, just their dataset generator.

Deliverables:
1. A `train_moons.py` script that uses **your** `MLP`, does gradient descent for 500 steps, and prints test accuracy every 50 steps.
2. A matplotlib plot of the decision boundary at step 0, 100, 500.
3. A short "what surprised me" paragraph in `SOLUTION_NOTES.md`.

Constraint: no NumPy vectorization inside the MLP. Every operation is a `Value`. This is slow. That's the point — you should feel the difference vs. vectorized NumPy in W4.
