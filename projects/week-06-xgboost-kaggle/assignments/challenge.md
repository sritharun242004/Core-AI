# Challenge — leakage policy for grouped and temporal features (3+ hours)

Design and implement one safe extension, with a short design note before code:

- **Ticket groups:** create a `TicketGroupSize` feature, but specify whether it is computed from training rows only, from a frozen reference table, or from a production-time group service. Test the policy with a validation-only ticket value.
- **Nested model selection:** use an inner CV loop to choose `max_depth` and `learning_rate`, then report an untouched outer ROC-AUC. Explain why selecting on the outer folds is leakage.
- **Calibration:** fit a calibration layer on a validation split and report Brier score plus ROC-AUC. Do not calibrate on the test CSV.

Required artifacts:

1. an ADR-style Markdown note stating the information boundary;
2. at least three failing tests written before implementation;
3. a deterministic experiment table with seed, folds, parameters, AUC, and runtime;
4. a schema test proving the original test row order and ID values survive.

A challenge is successful even if the new feature does not improve the metric. Honest negative results are more useful than a leaked score.
