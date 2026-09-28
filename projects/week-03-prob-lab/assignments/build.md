# Build (2-3 hrs)

Implement a coin-bias inference notebook.

1. Simulate 100 coin flips from a coin with true bias p = 0.65.
2. Start with a Beta(2, 2) prior.
3. After every 10 flips, plot the posterior Beta density on the same axes. You should see it tighten around 0.65 as data accumulates.
4. Add a KL-vs-true-posterior overlay: at every step, compute KL between the current posterior and the "final" posterior (from all 100 flips). It should decay monotonically.

Deliverables:
- `notebooks/coin_bias.ipynb` (or `.py` in percent format).
- A short "what surprised me" paragraph in `SOLUTION_NOTES.md`.
