# Solution notes — Week 6

## 1. Why the transformer is inside the pipeline

A median and a category vocabulary are statistics learned from data. If either is computed before cross-validation, information from the validation fold crosses the boundary. The result may look like a small implementation convenience, but it makes the reported ROC-AUC optimistic. `TitanicFeatures` is row-local and can run first; `ColumnTransformer` and its imputers are then fit anew in every fold by `cross_val_score`.

The custom `_PipelineWithDynamicPreprocessor` exists because arbitrary user CSVs may contain arbitrary extra columns, and feature engineering adds columns. It runs the row-local transformer once to discover post-engineering dtypes, creates the numeric/categorical `ColumnTransformer`, then delegates fitting to scikit-learn's normal `Pipeline`. The fitted object still exposes `named_steps["preprocess"]`, which makes the fold-local behavior inspectable.

## 2. Feature choices

- `FamilySize = SibSp + Parch + 1` captures the traveling party.
- `IsAlone` is a cheap nonlinear split.
- `FarePerPerson` avoids treating one group fare as the fare paid by every traveler.
- `Title` maps standard titles to four frequent groups and maps the long tail to `Rare`.
- `Deck` keeps the first cabin letter while discarding the high-cardinality cabin number.
- `CabinKnown` keeps the missingness signal.
- `PassengerId`, raw names, raw tickets, and raw cabin strings are excluded. Raw identifiers are usually memorization opportunities, and text columns have better compact representations in this small reference project.

None of these transforms compute a dataset-level statistic. `Ticket` group size would be tempting, but computing it across concatenated train and test frames would need an explicit leakage and deployment policy; it is left as a challenge extension.

## 3. Baseline versus boosted trees

The baseline is `LogisticRegression` over one-hot categories and imputed numeric values. Its score is a useful floor and a diagnostic: if a complex model cannot beat it, inspect data, folds, or calibration before adding complexity.

The XGBoost configuration is deliberately modest: histogram splits, 180 shallow trees, a small learning rate, row/column subsampling, and one thread. The synthetic fixture creates a diagonal class/sex interaction. It is nonlinear by construction, so the test asks XGBoost to beat logistic regression by 0.05 absolute ROC-AUC. This benchmark is deterministic and does not claim that XGBoost always improves the public Titanic score.

## 4. Missing values and unseen categories

Numeric columns use training-fold median imputation. An all-missing numeric column remains present (`keep_empty_features=True`) and receives the library's empty-feature fallback. Categorical values receive a constant `__missing__` token and are one-hot encoded with `handle_unknown="ignore"`; a category that appears only in test is therefore ignored rather than crashing or changing training dimensions.

Pandas nullable values are normalized to ordinary `numpy.nan` at the transformer boundary because scikit-learn's estimators accept the latter consistently across supported versions.

## 5. Submission contract

Kaggle's Titanic submission is intentionally small: the original test `PassengerId` in original row order and an integer `Survived` prediction. `make_submission` does not sort, reset, or merge the test frame. A duplicate or null ID is rejected before prediction. A path can be nested under a new directory, and the CLI additionally prevents overwriting the user-provided train/test input.

## 6. What surprised me

1. “Fit preprocessing inside CV” means the whole pipeline—not only the estimator—must be passed to `cross_val_score`.
2. A one-hot logistic model can approximate some interactions by ranking extreme categories, so a nonlinear fixture needs a genuinely non-additive target surface.
3. A schema-correct submission is a separate acceptance criterion from a good metric. Both deserve tests.
4. Unknown categories are a normal production event, not an exceptional test-only nuisance. `handle_unknown="ignore"` is part of the serving contract.

## 7. Safe extensions

- Add a nested CV layer for hyperparameter selection and keep the outer folds untouched.
- Compare calibrated probabilities and choose a threshold using a validation-only policy.
- Add permutation importance on an untouched validation set.
- Add a `TicketGroupSize` transformer that defines exactly which rows are allowed to share group statistics.
- Compare XGBoost against `RandomForestClassifier` and an RBF SVM while reporting fit time and peak memory.
