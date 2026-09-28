# Compute notes — Week 6

**Tier:** 🟢 green / local CPU  
**Expected hardware:** Apple Silicon MacBook or a normal laptop CPU  
**Network/auth:** none; no Kaggle credentials and no downloads are required  
**Fixture:** 900 train rows + 180 test rows, generated from a fixed seed  
**Typical run:** under one minute for the 23-test suite on a laptop

## Reproduce locally

```bash
uv pip install --python ../../.venv/bin/python scikit-learn xgboost pandas matplotlib
PYTHONPATH=src ../../.venv/bin/python -m pytest -q
```

The model uses `tree_method="hist"`, 180 shallow estimators, and `n_jobs=1`. That choice prioritizes a repeatable teaching run over maximum CPU throughput. If you increase `n_estimators`, `max_depth`, or CV folds, record the change beside the metric; do not compare numbers from different folds or fixture seeds as if they were one experiment.

## Optional real-data run

Download Titanic CSV files yourself only if you want a leaderboard experiment, then pass their local paths to the CLI:

```bash
PYTHONPATH=src ../../.venv/bin/python -m xgboost_kaggle \
  --train data/train.csv --test data/test.csv \
  --output artifacts/submission.csv
```

The repository does not include or fetch the dataset. Do not put credentials in the project, and do not treat a public leaderboard score as a substitute for a held-out evaluation.

## Budget and teardown

Estimated cost is **$0** on local hardware. There is no cloud runbook because this project does not need a GPU, persistent service, or remote instance. If you use a hosted notebook, stop/delete the runtime after the run and remove any uploaded CSVs from the provider workspace.
