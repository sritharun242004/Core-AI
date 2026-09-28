# prob-lab — Week 3 reference project

Monte Carlo estimators, Bayes-with-conjugate priors, and Shannon information (entropy, KL, cross-entropy) — all zero-safe.

## Run

```bash
cd projects/week-03-prob-lab
uv sync --extra dev
uv run pytest -v
```

## Public API

- `estimate_pi(n, seed)` — 4 × fraction of unit-square samples inside the unit quarter-circle.
- `expectation(fn, sampler, n)` — generic Monte Carlo mean.
- `posterior_mean(a, b, heads, tails)` — Beta-Binomial closed-form posterior mean.
- `log_bayes_factor(...)` — log ratio of hypothesis likelihoods + priors.
- `entropy(p)` / `kl_divergence(p, q)` / `cross_entropy(p, q)` — all in nats, all zero-safe.
