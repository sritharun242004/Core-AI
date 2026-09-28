# Warmup — inspect a tabular pipeline (30 minutes)

1. Run the offline test suite and read one fold's fitted `named_steps["preprocess"]`.
2. Call `make_synthetic_titanic(seed=7)` and report row counts, missing-value counts, class balance, and the dtypes of the raw columns.
3. Fit the baseline with three stratified folds. Explain why ROC-AUC is a ranking metric rather than an accuracy-at-0.5 metric.
4. Use `TitanicFeatures` on a three-row frame and verify that the input frame is unchanged.
5. Change the fixture seed. Record the new two mean AUCs, but do not call the new number a regression or improvement until you use the same seed and folds.

**Deliverable:** a short Markdown note with one table and one paragraph explaining which statistics are learned by the imputer and encoder.
