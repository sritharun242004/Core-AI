# Week 20 — Evaluation and MLOps pipeline

An offline, deterministic release gate: immutable examples, content-addressed datasets, seeded splits, exact match, counterbalanced judge callbacks, paired bootstrap intervals, training-bin drift, n-gram contamination diagnostics, trajectory metrics, and atomic JSONL results. Nothing here claims to reproduce a public benchmark score.

## Run

```bash
cd projects/week-20-evals-mlops-pipeline
uv run --extra dev pytest
uv run python notebooks/01-evaluation-pipeline.py
uv run ruff check src tests notebooks
```

With the repository's existing environment, use `PYTHONPATH="$PWD/src" ../../.venv/bin/python -m pytest tests`. NumPy is the only runtime dependency. Tests and notebooks do not use external APIs.

## Contracts

- `Example` is immutable; `dataset_digest` rejects duplicate IDs and hashes all fields independently of row order.
- `split_dataset` is a seeded random train/validation/test split. It is not appropriate for correlated users or temporal prediction without adapting the split policy.
- `judge_pair` calls a supplied judge twice with swapped answers. Contradictory votes are flagged, not silently treated as agreement. It measures preference under a rubric, not truth.
- `bootstrap_interval` resamples independent observations. For a paired release comparison, resample case-level differences, as `regression_gate` does. Repeated outputs for one prompt must be clustered before interpreting uncertainty.
- `population_stability` fits quantile edges on reference data and keeps open-ended tails. Smoothing prevents division by zero. PSI is a diagnostic, not a universal retraining threshold.
- `contamination` checks distinct prompt n-grams; shared templates can trigger it, semantic overlap can evade it, and unknown pretraining data remains unknown.
- `write_results` serializes before replacing the old report. Nonfinite JSON values and duplicate IDs fail without corrupting existing results.

## Optional ecosystem integrations

`evals_mlops.integrations.export_mlflow` lazily imports MLflow and writes to a caller-selected **local file URI**. `inspect_task` constructs a real Inspect AI task from the fixture; executing it requires an explicitly configured model. These optional packages are not installed or executed by the core tests.

- MLflow: install a compatible version in a separate environment, call `export_mlflow(manifest, metrics, './mlruns')`, inspect locally. Do not log sensitive prompts.
- DVC: `dvc init --no-scm`, `dvc add data/evaluation.jsonl`; preserve the generated checksum metadata alongside the release manifest. A remote is optional; do not upload secrets or restricted datasets.
- W&B: use `WANDB_MODE=offline` for an explicitly installed client and retain the offline run directory. Syncing is a separate consent/budget decision, never done here.
- Inspect AI is open-source UK AI Security Institute tooling. MT-Bench is a multi-turn model-judge methodology; HELM is a broad evaluation framework; MMLU-Pro tests harder knowledge/reasoning; SWE-Bench evaluates repository issue resolution; Harbor runs agent evaluation tasks in controlled environments. This tiny harness substitutes for none of their datasets, scorers, or isolation requirements.

Connect W8 `ml_eval_suite` for tabular baselines, W13 `nano_gpt_ssm` for causal LM outputs, and W15a `post_training_lab` for preference-policy comparisons. Evaluate the **same held-out case IDs**, preserve tokenizer/prompt/model revisions, and avoid tuning on the final test set.
