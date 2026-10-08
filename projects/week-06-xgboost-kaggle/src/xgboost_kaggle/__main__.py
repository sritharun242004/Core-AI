"""Command-line entry point for an offline user-CSV submission run."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import evaluate_models, fit_model, load_titanic, make_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost on user-provided Titanic CSVs")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", default=Path("submission.csv"), type=Path)
    parser.add_argument("--model", choices=("baseline", "xgboost"), default="xgboost")
    parser.add_argument("--folds", default=5, type=int)
    args = parser.parse_args()

    input_paths = {args.train.resolve(), args.test.resolve()}
    if args.output.resolve() in input_paths:
        parser.error("output would overwrite a user input CSV; choose another output path")
    train, test = load_titanic(args.train, args.test)
    scores = evaluate_models(train, n_splits=args.folds, seed=42)
    print(f"user-provided CSVs: {len(train)} train rows, {len(test)} test rows")
    print(
        f"roc_auc baseline={scores['baseline'].mean():.4f} xgboost={scores['xgboost'].mean():.4f}"
    )
    model = fit_model(train, model=args.model, seed=42)
    make_submission(model, test, args.output)
    print(f"wrote {args.output} with PassengerId, Survived columns")


if __name__ == "__main__":
    main()
