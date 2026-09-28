# Solution notes — prob-lab

- **Zero-safe entropy**: mask `p > 0` before computing `log`. Never add ε to probabilities — that biases the estimate. The 0·log(0) = 0 convention is safe because we're skipping those terms entirely.
- **KL divergence** returns `inf` when `q_i = 0 and p_i > 0`. That's mathematically correct — the reverse KL is finite there, though; know the difference.
- **Beta-Binomial conjugacy**: (α, β) + (h, t) → (α+h, β+t). This is why Bayesian analysts love the Beta family for binary outcomes.
- **π by Monte Carlo** converges at rate O(1/√n). At n=200k the standard error is ≈ 0.004; the test allows 0.02 to avoid flakiness.
- **Bayes factor pitfall**: the log ratio can be a huge number. Never exponentiate before comparing — stay in log-space.
