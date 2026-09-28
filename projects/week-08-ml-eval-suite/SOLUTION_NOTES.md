# Solution notes

## Why out-of-fold calibration matters

Fitting an estimator on all rows and calibrating those same predictions measures memorization and gives an overconfident curve. `calibration_curve_data` uses `cross_val_predict(..., method="predict_proba")`, so every probability is produced by a model that did not train on that row.

## Leakage detector scope

The detector is an inexpensive smoke test: it reports per-feature correlations with each target and flags a feature that crosses the threshold in both splits. It cannot discover every temporal, duplicate-row, preprocessing, or label-proxy leak. A green report is an invitation to inspect the data—not a certificate.

## Reproducibility

The CV function delegates to `cross_val_score` directly, with one worker. Keep the estimator’s random state fixed when the estimator is stochastic, and keep preprocessing inside a `Pipeline` so each fold learns transformations from training rows only.
