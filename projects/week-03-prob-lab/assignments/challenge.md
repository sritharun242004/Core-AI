# Challenge (3+ hrs)

Implement **importance sampling** and compare it to naive Monte Carlo on a heavy-tailed problem.

Setup: estimate E[X²] where X ~ N(0, 1) truncated to |x| > 3 (the tails).

1. Naive MC: sample from N(0, 1) and reject |x| ≤ 3. Rejection rate ≈ 99.7% — inefficient.
2. Importance sampling: draw from a Cauchy proposal q(x), weight by p(x)/q(x). Should converge with 10-100× fewer samples.
3. Plot: variance-vs-samples for both methods. IS should win by an order of magnitude.
4. Write it up in `SOLUTION_NOTES.md` with the two curves.

Stretch: extend to a 2-D truncated Gaussian and compare the two approaches on integrated squared error.
