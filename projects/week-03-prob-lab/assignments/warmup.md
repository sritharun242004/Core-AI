# Warmup (30 min)

1. Verify empirically that `estimate_pi`'s standard error scales like `1/√n`. Plot n vs |estimate - π| on a log-log axis for n ∈ {10², 10³, 10⁴, 10⁵}.
2. Add a `posterior_variance(prior_a, prior_b, heads, tails)` closed-form function. The Beta variance formula is `αβ / ((α+β)² (α+β+1))`.
3. Compute `entropy([0.5, 0.5])` and `entropy([0.9, 0.1])` — is the second smaller? By how much?
