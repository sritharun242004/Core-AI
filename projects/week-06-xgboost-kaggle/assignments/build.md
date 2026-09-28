# Build — make the submission pipeline yours (2–3 hours)

Extend the reference project without breaking its input contract:

1. Add an optional `RandomForestClassifier` model to `build_pipeline`.
2. Evaluate logistic regression, random forest, and XGBoost on the same `StratifiedKFold` indices. Return per-fold AUC, mean, standard deviation, and fit time.
3. Plot the three fold distributions. Label the chart with the fixture seed and fold count.
4. Add a validation-only threshold report: precision, recall, and confusion matrix at thresholds 0.25, 0.50, and 0.75. Do not select a threshold using the test CSV.
5. Add a test that a test-only category does not alter the fitted one-hot feature count.

**Acceptance:** the output CSV remains exactly `PassengerId,Survived`; no preprocessing is fit outside a fold; and the report clearly distinguishes mean AUC from the public Titanic leaderboard.
