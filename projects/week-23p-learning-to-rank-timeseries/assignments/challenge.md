# Challenge — 3+ hours

Choose one extension:

- Implement an explicit optional LambdaMART adapter with a pinned locally installed ranking library. Keep queries contiguous, pass correct group sizes, preserve heldout query boundaries, and compare against the neural lambdas under the same metric. Label tree and neural results separately.
- Add cluster randomization and a cluster-aware uncertainty estimator. Demonstrate why treating dependent page views as independent produces overconfident intervals. Specify a causal estimand, interference risks and missingness assumptions.
- Add a past-only forecasting validation/calibration slice and quantile or conformal intervals. Measure interval coverage and width by horizon. Compare with seasonal naive; explain why rolling-origin errors cannot automatically be treated as iid.

Include an ablation, a failure case and a written algorithm boundary. TFT/Prophet comparisons require real implementations and correct covariate availability; simply renaming the existing model does not count. Design guidance: `SOLUTION_NOTES.md`; these extensions remain learner work.
