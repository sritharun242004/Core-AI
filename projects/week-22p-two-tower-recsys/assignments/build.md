# Build — 2–3 hours

Run the notebook and tests offline. Compare popularity, ID towers and one-hop graph towers on the same full catalog and heldout users. Record k, positive-count denominator, macro averaging, tie break and seen exclusion.

Add tests that a holdout edge is absent from both graph directions, a cold-start user never indexes an unknown embedding, and duplicate events cannot cross splits. Inspect both embedding gradients. Introduce a separate validation slice before tuning dimensions or regularization; preserve an untouched final test slice and timestamp boundaries.

Optionally use a local MovieLens ratings.csv. Document the rating threshold, filtering and treatment of repeated interactions. Do not download data from the notebook or claim the synthetic result is MovieLens accuracy.

Deliver a reproducible report with baseline, per-user outcomes, losses and at least three limitations. Solutions and expected contracts: `SOLUTION_NOTES.md`.
