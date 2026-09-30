# Build — 2–3 hours

Run all tests and the notebook. Report pairwise and LambdaRank-style heldout grouped NDCG/Recall, not training accuracy alone. Verify whole-query split disjointness, masked-candidate denominators, ties and zero-pair groups.

Simulate an A/B treatment effect with independent randomization units. Report arm counts, SRM p-value, ATE, SE and a prespecified confidence interval. Show a realization where a valid interval misses the true effect; explain coverage without rerolling seeds to hide it. Deliberately introduce allocation loss and detect SRM.

Run expanding-window forecasts and compare per-origin MAE against seasonal naive. Mutate all values after one origin; predictions and fitted scaling parameters at that origin must remain unchanged, while scored errors should change. Inspect first-backcast and forecast-head gradients.

Deliver a compact three-part report with a clear separation between ranking quality, causal evidence and temporal prediction. Solutions: `SOLUTION_NOTES.md`.
