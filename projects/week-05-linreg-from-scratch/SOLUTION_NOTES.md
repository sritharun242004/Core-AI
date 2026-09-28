# Solution notes

## What to inspect

- Both solvers minimize mean half-MSE plus `l2 / 2 * ||coef||²`. The regularized normal equation therefore uses `n * l2`, not `l2`, on the feature diagonal. The closed-form path solves an augmented least-squares system with `lstsq` so rank-deficient features are safe.
- The intercept is stored as the first entry of an augmented design matrix but is excluded from the L2 penalty.
- `np.logaddexp(0, z) - y*z` computes logistic loss without first materializing `log(sigmoid(z))`.
- The sigmoid implementation uses separate positive and negative branches so a large negative logit does not overflow `exp(-z)`.
- Gradient descent uses backtracking only when a proposed step increases loss. This preserves a monotonic teaching curve without hiding the gradient.

## Common wrong turns

1. Regularizing the intercept changes the decision boundary for no principled reason.
2. Comparing a closed-form MSE objective with a gradient-descent objective that includes a differently scaled penalty gives apparently inconsistent coefficients.
3. Calling `exp` on raw logits can produce `inf`, then `nan` after normalization.
4. A low training loss is not evidence of a good classifier; use a held-out split and ROC-AUC in the challenge.
