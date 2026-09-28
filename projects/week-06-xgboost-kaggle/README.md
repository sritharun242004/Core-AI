# xgboost-kaggle — Week 6 reference project

An offline, leakage-safe Titanic-style binary-classification pipeline. It takes a user-provided `train.csv` and `test.csv`, validates their contract, performs row-local Titanic feature engineering, imputes mixed numeric/categorical missing values inside a scikit-learn `Pipeline`, compares a logistic baseline with XGBoost using stratified ROC-AUC cross-validation, and writes a Kaggle-compatible submission.

The project intentionally does **not** download Kaggle data, call a network API, or require Kaggle credentials. The test suite generates a deterministic nonlinear fixture. On that fixture, XGBoost is required to beat the additive linear baseline by at least **0.05 absolute ROC-AUC**. That is a controlled teaching fixture, not a promise about a real Titanic leaderboard.

## Run the tests

From this directory, with dependencies installed in the repository's existing virtual environment:

```bash
PYTHONPATH=src ../../.venv/bin/python -m pytest -q
```

Or install this project into an isolated environment with `uv sync --extra dev` and run `uv run pytest -q`. The package is named `xgboost-kaggle` and the importable module is `xgboost_kaggle`; the Hatch wheel configuration makes that mapping explicit.

## Run on user CSVs

```bash
PYTHONPATH=src ../../.venv/bin/python -m xgboost_kaggle \
  --train /path/to/train.csv \
  --test /path/to/test.csv \
  --output artifacts/submission.csv \
  --folds 5
```

The CLI prints baseline and XGBoost mean ROC-AUC, fits the selected model on all training rows, and writes exactly two columns in test-row order:

```text
PassengerId,Survived
892,0
893,1
```

Use `--model baseline` to make a logistic-regression submission for comparison. The CLI rejects an output path that would overwrite either input CSV.

## Input contract

`train.csv` must contain:

- `PassengerId`: non-null and unique;
- `Survived`: finite binary labels (`0` or `1`), with both classes present;
- `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`;
- optional standard Titanic columns such as `Name`, `Ticket`, and `Cabin`;
- arbitrary additional numeric or categorical columns are allowed.

`test.csv` has the same feature columns and `PassengerId`, but must not contain `Survived`. Known numeric columns may contain missing values; categorical columns may contain missing values and unseen test categories. IDs are preserved but never learned as features. Invalid labels, non-finite numeric values, negative count/fare values, duplicate IDs, missing required columns, and accidental test labels fail loudly rather than producing a questionable submission.

## Pipeline design

`src/xgboost_kaggle/pipeline.py` contains the public API:

- `TitanicFeatures` adds row-local `FamilySize`, `IsAlone`, `FarePerPerson`, `Title`, `Deck`, and `CabinKnown`; it never learns a global statistic and never mutates the input frame.
- `build_pipeline("baseline" | "xgboost")` returns a fresh scikit-learn pipeline. Numeric columns use median imputation; categorical columns use a missing sentinel and `OneHotEncoder(handle_unknown="ignore")`.
- `evaluate_models` creates a `StratifiedKFold` splitter and scores a fresh complete pipeline in each fold. Imputer medians and category vocabularies therefore come only from that fold's training rows.
- `fit_model` validates and fits on all training rows after model selection.
- `make_submission` thresholds positive-class probabilities and writes only `PassengerId,Survived`.
- `make_synthetic_titanic` is deterministic with a seed and is used for offline tests and notebook exploration.

The implementation uses one thread for XGBoost so local runs are reproducible and polite on a laptop. `tree_method="hist"` keeps the green-tier experiment fast without requiring a GPU.

## Why ROC-AUC and why a baseline?

ROC-AUC measures ranking quality across all classification thresholds, which avoids choosing a Titanic threshold before inspecting the operating point. The logistic model is intentionally simple: one-hot categories and additive coefficients. XGBoost builds additive trees whose split interactions can represent patterns such as “class 1 + female” without manually expanding every interaction. The useful comparison is the fold-wise score difference, not a single lucky holdout.

## Files

- `src/xgboost_kaggle/` — importable pipeline and CLI.
- `tests/test_pipeline.py` — 23 offline numerical, leakage, validation, model, submission, and CLI tests.
- `notebooks/01-offline-titanic-pipeline.py` — percent-format notebook for fixture EDA, CV, and submission inspection.
- `SOLUTION_NOTES.md` — implementation decisions and debugging notes.
- `COMPUTE.md` — local compute budget and reproducibility notes.
- `assignments/` — warmup, build, and challenge extensions.

## Limitations and responsible interpretation

This is a reference pipeline, not an official Kaggle solution. Public Titanic data can have different quirks, feature distributions, and leaderboard behavior. Do not infer a company's undocumented production architecture from this project. The point is to learn a defensible tabular workflow: validate inputs, fit transformations inside CV, compare against a baseline, and publish a schema-correct artifact.
