"""Leakage-safe Titanic-style preprocessing, CV, fitting, and submission helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self, cast, overload

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from ._sklearn import FloatArray, PipelineFit, PredictProba, cross_val_score

ID_COLUMN = "PassengerId"
TARGET_COLUMN = "Survived"
REQUIRED_FEATURES = ("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")
NONNEGATIVE_COLUMNS = ("Pclass", "Age", "SibSp", "Parch", "Fare")


class TitanicFeatures(BaseEstimator, TransformerMixin):
    """Create row-local Titanic features while retaining arbitrary user columns.

    The transformer deliberately learns no statistics. It can therefore sit as
    the first step inside a scikit-learn Pipeline and be fitted independently
    in every CV training fold.
    """

    def fit(self, x: object, y: object = None) -> TitanicFeatures:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("TitanicFeatures expects a pandas DataFrame")
        self.input_columns_ = tuple(x.columns)
        return self

    @staticmethod
    def _title(name: Any) -> Any:
        if pd.isna(name):
            return pd.NA
        text = str(name)
        if "," not in text or "." not in text:
            return "Rare"
        title = text.split(",", 1)[1].split(".", 1)[0].strip()
        common = {"Mr", "Mrs", "Miss", "Master"}
        return title if title in common else "Rare"

    @staticmethod
    def _deck(cabin: Any) -> Any:
        if pd.isna(cabin):
            return pd.NA
        text = str(cabin).strip()
        return text[0].upper() if text else pd.NA

    def transform(self, x: object) -> pd.DataFrame:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("TitanicFeatures expects a pandas DataFrame")
        frame = x.copy(deep=True)
        if {"SibSp", "Parch"}.issubset(frame.columns):
            # min_count=2 preserves missingness instead of silently turning a
            # missing count into zero.
            frame["FamilySize"] = frame[["SibSp", "Parch"]].sum(axis=1, min_count=2) + 1
            frame["IsAlone"] = (frame["FamilySize"] == 1).astype("Int64")
        if {"Fare", "FamilySize"}.issubset(frame.columns):
            frame["FarePerPerson"] = frame["Fare"] / frame["FamilySize"]
        if "Name" in frame.columns:
            frame["Title"] = frame["Name"].map(self._title).astype("string")
        if "Cabin" in frame.columns:
            frame["Deck"] = frame["Cabin"].map(self._deck).astype("string")
            frame["CabinKnown"] = frame["Cabin"].notna().astype("Int64")
        # IDs and raw high-cardinality text are not model features. Their
        # useful information is represented by the row-local features above.
        drop_columns = [
            column
            for column in (ID_COLUMN, TARGET_COLUMN, "Name", "Ticket", "Cabin")
            if column in frame
        ]
        frame.drop(columns=drop_columns, inplace=True)
        # sklearn's imputers accept ordinary object/string columns reliably;
        # normalize pandas' nullable scalar to NumPy NaN at the pipeline edge.
        for column in frame.select_dtypes(include=["string", "object"]).columns:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
        return frame


def _validate_columns(frame: object, *, training: bool) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("CSV data must be loaded into a pandas DataFrame")
    required = (ID_COLUMN, TARGET_COLUMN) if training else (ID_COLUMN,)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")
    feature_missing = [column for column in REQUIRED_FEATURES if column not in frame.columns]
    if feature_missing:
        raise ValueError(f"missing required feature(s): {', '.join(feature_missing)}")
    ids = frame[ID_COLUMN]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("PassengerId must be non-null and unique")
    for column in NONNEGATIVE_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().ne(frame[column].isna()).any():
            raise ValueError(f"{column} must be numeric")
        if np.isinf(numeric.dropna()).any():
            raise ValueError(f"{column} must contain only finite values")
        if (numeric.dropna() < 0).any():
            raise ValueError(f"{column} must be nonnegative")
    if training:
        target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        if target.isna().any() or not np.isfinite(target).all():
            raise ValueError("Survived must contain finite binary labels")
        if not target.isin([0, 1]).all():
            raise ValueError("Survived must contain only 0 or 1")
        if target.nunique() < 2:
            raise ValueError("Survived must contain both classes")


def _make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in x.columns if column not in numeric]
    if not numeric and not categorical:
        raise ValueError("no usable model features")
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
    )
    categorical_pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__missing__",
                    keep_empty_features=True,
                ),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model: str = "xgboost", seed: int = 0) -> _PipelineWithDynamicPreprocessor:
    """Build a fresh estimator; preprocessing is fitted by Pipeline.fit."""
    if model == "baseline":
        estimator = LogisticRegression(max_iter=1_000, solver="liblinear", random_state=seed)
    elif model == "xgboost":
        estimator = XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=seed,
            tree_method="hist",
        )
    else:
        raise ValueError("model must be 'baseline' or 'xgboost'")

    # The first transformer is fitted on the fold's X, then the preprocessor
    # is built from that fold's schema by _PipelineWithDynamicPreprocessor.
    return _PipelineWithDynamicPreprocessor(
        [
            ("features", TitanicFeatures()),
            ("preprocess", "dynamic"),
            ("model", estimator),
        ]
    )


class _PipelineWithDynamicPreprocessor(Pipeline):
    """Pipeline that selects dtypes after row-local feature engineering.

    ColumnTransformer needs the post-feature-engineering column lists. Building
    them in fit keeps all imputation and category discovery fold-local while
    retaining the familiar ``named_steps['preprocess']`` inspection surface.
    """

    @overload
    def fit(self, X: object, y: object = None, **fit_params: object) -> Self:  # noqa: N803
        ...

    @overload
    def fit(self, *, x: pd.DataFrame, y: object = None, **fit_params: object) -> Self: ...

    def fit(self, X: object = None, y: object = None, **fit_params: object) -> Self:  # noqa: N803
        # sklearn names the input X; retain the original public x= call as well.
        x = fit_params.pop("x", X)
        if not isinstance(x, pd.DataFrame):
            raise TypeError("pipeline expects a pandas DataFrame")
        features = self.steps[0][1]
        transformed = features.fit_transform(x, y)
        self.steps[1] = ("preprocess", _make_preprocessor(transformed))
        cast(PipelineFit, super().fit)(x, y, **fit_params)
        return self

    def predict_proba(self, X: object, **predict_proba_params: object) -> FloatArray:  # noqa: N803
        # The parent accepts array-like objects; no new runtime restriction is added.
        return cast(PredictProba, super().predict_proba)(
            cast(pd.DataFrame, X), **predict_proba_params
        )


def validate_train(train: pd.DataFrame) -> None:
    _validate_columns(train, training=True)


def validate_test(test: pd.DataFrame) -> None:
    _validate_columns(test, training=False)
    if TARGET_COLUMN in test.columns:
        raise ValueError("test data must not contain Survived")


def load_titanic(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load user-provided CSVs without network access or Kaggle credentials."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    validate_train(train)
    validate_test(test)
    return train, test


def fit_model(
    train: pd.DataFrame, model: str = "xgboost", seed: int = 0
) -> _PipelineWithDynamicPreprocessor:
    validate_train(train)
    estimator = build_pipeline(model, seed=seed)
    estimator.fit(train.drop(columns=TARGET_COLUMN), train[TARGET_COLUMN].astype(int))
    return estimator


def evaluate_models(
    train: pd.DataFrame, n_splits: object = 5, seed: int = 0
) -> dict[str, FloatArray]:
    validate_train(train)
    if not isinstance(n_splits, int) or n_splits < 2 or n_splits > len(train):
        raise ValueError("n_splits must be an integer between 2 and the number of rows")
    x = train.drop(columns=TARGET_COLUMN)
    y = train[TARGET_COLUMN].astype(int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return {
        name: cross_val_score(
            build_pipeline(name, seed=seed),
            x,
            y,
            cv=cv,
            scoring="roc_auc",
            error_score="raise",
        )
        for name in ("baseline", "xgboost")
    }


def make_submission(
    model: Pipeline,
    test: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Predict in test-row order and optionally write Kaggle's exact schema."""
    validate_test(test)
    probabilities = cast(PredictProba, model.predict_proba)(test)[:, 1]
    submission = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: (probabilities >= 0.5).astype(int),
        }
    )
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(destination, index=False)
    return submission


def make_synthetic_titanic(
    n_train: int = 900,
    n_test: int = 180,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic, mixed-type nonlinear fixture without downloads.

    The target contains class/sex and age/fare interactions, giving trees a
    meaningful advantage over an additive logistic baseline. It is a test
    fixture, not a claim about the public Titanic leaderboard.
    """
    if n_train < 20 or n_test < 1:
        raise ValueError("n_train must be at least 20 and n_test must be positive")
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    pclass = rng.choice([1, 2, 3], n_total, p=[0.22, 0.28, 0.50])
    sex = rng.choice(["female", "male"], n_total, p=[0.38, 0.62]).astype(object)
    age = np.clip(rng.normal(31, 13, n_total), 0.4, 78).astype(float)
    fare = np.maximum(4, rng.lognormal(3.0, 0.65, n_total) / pclass)
    sibsp = rng.poisson(0.5, n_total).astype(float)
    parch = rng.poisson(0.35, n_total).astype(float)
    embarked = rng.choice(["S", "C", "Q"], n_total, p=[0.70, 0.19, 0.11]).astype(object)
    names = np.where(sex == "female", "Doe, Mrs. Ada", "Doe, Mr. Bob").astype(object)
    cabins = np.where(
        rng.random(n_total) < 0.28,
        rng.choice(["C85", "E12", "B20"], n_total),
        np.array(None, dtype=object),
    ).astype(object)
    tickets = np.array(
        [f"T{value:04d}" for value in rng.integers(1000, 9999, n_total)],
        dtype=object,
    )
    missing = rng.random(n_total)
    age[missing < 0.08] = np.nan
    fare[missing > 0.94] = np.nan
    sex[missing > 0.975] = None
    embarked[(missing > 0.95) & (missing <= 0.975)] = None
    cabins[missing < 0.16] = None
    # Nonlinear interaction signal: the two diagonal class/sex groups survive
    # more often, while the other two groups do not. An additive linear model
    # cannot represent this XOR-like surface; boosted trees can.
    good_group = ((pclass == 1) & (sex == "female")) | ((pclass == 3) & (sex == "male"))
    interaction = (
        4.2 * good_group
        - 2.4
        + 0.55 * ((age < 16) & (pclass == 3))
        + 0.08 * np.nan_to_num(fare, nan=np.nanmedian(fare))
    )
    probability = 1 / (1 + np.exp(-interaction))
    labels = rng.binomial(1, probability).astype(int)
    ids = np.arange(1, n_total + 1)
    frame = pd.DataFrame(
        {
            ID_COLUMN: ids,
            "Pclass": pclass,
            "Name": names,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Ticket": tickets,
            "Fare": fare,
            "Cabin": cabins,
            "Embarked": embarked,
            TARGET_COLUMN: labels,
        }
    )
    train = frame.iloc[:n_train].reset_index(drop=True)
    test = frame.iloc[n_train:].drop(columns=TARGET_COLUMN).reset_index(drop=True)
    return train, test
