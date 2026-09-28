"""Offline numerical and CSV-contract tests; no Kaggle downloads or credentials."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from xgboost_kaggle import (
    TitanicFeatures,
    build_pipeline,
    evaluate_models,
    fit_model,
    load_titanic,
    make_submission,
    make_synthetic_titanic,
)


@pytest.fixture(scope="module")
def data():
    return make_synthetic_titanic(n_train=900, n_test=180, seed=42)


@pytest.fixture(scope="module")
def scores(data):
    train, _ = data
    return evaluate_models(train, n_splits=3, seed=42)


def test_fixture_is_deterministic_mixed_and_unlabelled_at_test_time(data):
    train, test = data
    again_train, again_test = make_synthetic_titanic(900, 180, seed=42)
    pd.testing.assert_frame_equal(train, again_train)
    pd.testing.assert_frame_equal(test, again_test)
    assert set(train.Survived) == {0, 1}
    assert "Survived" not in test
    assert train.Age.isna().any() and train.Sex.isna().any()
    assert set(train.PassengerId).isdisjoint(test.PassengerId)


def test_xgboost_beats_linear_baseline_on_nonlinear_fixture(scores):
    # Absolute AUC difference, not a percent-relative or real-Titanic guarantee.
    assert set(scores) == {"baseline", "xgboost"}
    for values in scores.values():
        assert values.shape == (3,)
        assert np.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()
    assert scores["xgboost"].mean() - scores["baseline"].mean() >= 0.05
    assert scores["xgboost"].mean() >= 0.80


def test_scores_match_sklearn_on_identical_stratified_folds(data, scores):
    train, _ = data
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    expected = cross_val_score(
        build_pipeline("baseline", seed=42),
        train.drop(columns="Survived"),
        train.Survived,
        cv=cv,
        scoring="roc_auc",
        error_score="raise",
    )
    np.testing.assert_allclose(scores["baseline"], expected, atol=1e-12)


def test_feature_engineering_is_row_local_and_does_not_mutate_inputs(data):
    train, _ = data
    frame = train.iloc[:3].copy()
    frame["SibSp"] = [1.0, 0.0, np.nan]
    frame["Parch"] = [2.0, 0.0, 0.0]
    frame["Fare"] = [80.0, 20.0, np.nan]
    frame["Name"] = ["Smith, Mrs. Ada", "Lee, Dr. Sam", None]
    frame["Cabin"] = ["C85", None, " E12"]
    original = frame.copy(deep=True)
    transformed = TitanicFeatures().fit_transform(frame)
    np.testing.assert_allclose(transformed.FamilySize, [4, 1, np.nan], equal_nan=True)
    np.testing.assert_allclose(transformed.FarePerPerson, [20, 20, np.nan], equal_nan=True)
    assert transformed.Title.iloc[:2].tolist() == ["Mrs", "Rare"]
    assert transformed.Title.isna().iloc[2]
    assert transformed.Deck.iloc[[0, 2]].tolist() == ["C", "E"]
    assert not {"PassengerId", "Survived", "Name", "Ticket", "Cabin"} & set(transformed)
    pd.testing.assert_frame_equal(frame, original)


def test_preprocessing_is_fitted_only_on_training_fold(data):
    train, _ = data
    frame = train.iloc[:90].drop(columns="Survived").copy()
    frame["Age"] = np.r_[np.arange(60, dtype=float), np.full(30, 1000.0)]
    frame.loc[frame.index[:5], "Age"] = np.nan
    frame["Embarked"] = ["S"] * 60 + ["validation-only"] * 30
    y = np.tile([0, 1], 45)
    results = cross_validate(
        build_pipeline("baseline"),
        frame,
        y,
        cv=[(np.arange(60), np.arange(60, 90))],
        scoring="roc_auc",
        return_estimator=True,
        error_score="raise",
    )
    fitted = results["estimator"][0]
    assert isinstance(fitted, Pipeline)
    preprocess = fitted.named_steps["preprocess"]
    numeric = preprocess.named_transformers_["numeric"]
    numeric_columns = preprocess.transformers_[0][2]
    age_index = list(numeric_columns).index("Age")
    assert numeric.named_steps["imputer"].statistics_[age_index] == 32.0
    assert frame.Age.median() != 32.0
    encoder = preprocess.named_transformers_["categorical"].named_steps["encoder"]
    assert all("validation-only" not in categories for categories in encoder.categories_)


@pytest.mark.parametrize("model_name", ["baseline", "xgboost"])
def test_pipeline_handles_all_missing_columns_and_unknown_categories(data, model_name):
    train, test = (frame.copy() for frame in data)
    train["Age"] = np.nan
    train["Cabin"] = None
    train["Embarked"] = pd.NA
    train["Sex"] = train.Sex.astype("string")
    test["Embarked"] = "new-port"
    test["Cabin"] = "Z999"
    model = fit_model(train, model=model_name)
    proba = model.predict_proba(test)
    assert proba.shape == (len(test), 2)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-6)
    changed_ids = test.assign(PassengerId=test.PassengerId + 100_000)
    np.testing.assert_allclose(proba, model.predict_proba(changed_ids), atol=1e-12)


def test_submission_round_trip_has_exact_schema_and_preserves_order(data, tmp_path):
    train, test = data
    test = test.iloc[::-1].copy()
    model = fit_model(train)
    output = tmp_path / "nested" / "submission.csv"
    submission = make_submission(model, test, output)
    assert output.is_file()
    restored = pd.read_csv(output)
    pd.testing.assert_frame_equal(submission, restored)
    assert restored.columns.tolist() == ["PassengerId", "Survived"]
    assert restored.PassengerId.tolist() == test.PassengerId.tolist()
    assert set(restored.Survived) <= {0, 1}
    assert restored.Survived.dtype.kind in "iu"
    assert len(restored) == len(test)


def test_user_csv_loading_and_optional_text_columns(data, tmp_path):
    train, test = (frame.drop(columns=["Name", "Cabin", "Ticket"]) for frame in data)
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    loaded_train, loaded_test = load_titanic(train_path, test_path)
    assert loaded_train.shape == train.shape
    submission = make_submission(fit_model(loaded_train, model="baseline"), loaded_test)
    assert submission.PassengerId.tolist() == test.PassengerId.tolist()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Survived", 2, "Survived"),
        ("Survived", np.nan, "Survived"),
        ("PassengerId", 1, "PassengerId"),
        ("PassengerId", np.nan, "PassengerId"),
        ("Age", np.inf, "finite"),
        ("Age", "not-a-number", "numeric"),
        ("SibSp", -2, "nonnegative"),
    ],
)
def test_invalid_training_data_is_rejected(data, column, value, message):
    train = data[0].copy()
    train[column] = value
    with pytest.raises(ValueError, match=message):
        fit_model(train)


def test_missing_required_features_and_single_class_are_rejected(data):
    train, _ = data
    with pytest.raises(ValueError, match="Fare"):
        fit_model(train.drop(columns="Fare"))
    with pytest.raises(ValueError, match="both"):
        fit_model(train.assign(Survived=0))


@pytest.mark.parametrize("folds", [1, 2.5, 1000])
def test_invalid_cv_fold_count_is_rejected(data, folds):
    with pytest.raises(ValueError, match="n_splits"):
        evaluate_models(data[0], n_splits=folds)


def test_submission_rejects_labels_and_duplicate_ids(data):
    train, test = data
    model = fit_model(train, model="baseline")
    with pytest.raises(ValueError, match="Survived"):
        make_submission(model, test.assign(Survived=0))
    with pytest.raises(ValueError, match="PassengerId"):
        make_submission(model, test.assign(PassengerId=1))


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="model"):
        build_pipeline("imaginary")


def test_cli_accepts_user_csv_and_reports_auc(data, tmp_path):
    train, test = data
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    output = tmp_path / "submission.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    command = [
        sys.executable, "-m", "xgboost_kaggle", "--train", str(train_path),
        "--test", str(test_path), "--output", str(output), "--folds", "3",
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env, check=True)
    assert "roc_auc" in result.stdout
    assert "user-provided" in result.stdout
    assert pd.read_csv(output).columns.tolist() == ["PassengerId", "Survived"]

    # A typo must never overwrite a user's input dataset.
    command[command.index("--output") + 1] = str(train_path)
    rejected = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    assert rejected.returncode != 0
    assert "overwrite" in rejected.stderr
    pd.testing.assert_frame_equal(pd.read_csv(train_path), pd.read_csv(train_path))
