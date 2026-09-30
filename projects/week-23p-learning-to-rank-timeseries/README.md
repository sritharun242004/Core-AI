# Track P β — Ranking, experiments and forecasts

Three small but real implementations connect offline ranking to product evidence and temporal prediction:

1. A **neural pairwise reranker** trained on grouped queries, with RankNet and frozen-rank **LambdaRank-style delta-NDCG gradients**.
2. Stable unit-level randomization, difference-in-means ATE, Welch standard error, normal confidence intervals and a sample-ratio-mismatch (SRM) test.
3. Rolling-origin forecasting with a seasonal-naive baseline and an actual tiny **N-BEATS-like residual backcast/forecast stack**.

All fixtures are synthetic. No network, credentials, downloads, cloud, external ranking package or cross-project imports are required.

## Run (Python 3.13)

From the repository root using the existing environment:

```bash
ROOT="$PWD"
cd projects/week-23p-learning-to-rank-timeseries
PYTHONPATH="$PWD/src" "$ROOT/.venv/bin/python" -m pytest
PYTHONPATH="$PWD/src" "$ROOT/.venv/bin/python" notebooks/01-ranking-experiments-forecast.py
"$ROOT/.venv/bin/ruff" check src tests notebooks
```

The percent-format notebook runs as plain Python and can be opened/converted with Jupytext in your own environment. Standalone users may install the package and its `dev` extra in an isolated Python 3.13 environment.

## Ranking API and contracts

- `ranking_fixture()`: 30 queries × 6 documents with feature-derived graded relevance.
- `group_split(groups)`: disjoint whole-query train/holdout indices; never randomly split documents within a query.
- `NeuralRanker(features)`: actual two-layer tanh MLP; final score bias omitted because pairwise margins cannot identify it.
- `pairwise_loss(scores, relevance, groups, method='pairwise'|'lambda', k=10)`: strictly preferred within-query pairs only; query-wise means then macro mean over all queries, including zero-pair queries.
- `lambda_gradients(...)`: independently implemented analytical score gradients of the frozen-rank delta-NDCG surrogate. Tests compare these with autograd and finite differences away from rank changes.
- `train_ranker(...)`: Adam updates on training groups only; full loss history.
- `grouped_metrics(..., eligible=mask)`: macro Recall/NDCG, zero for empty relevance, stable input-row tie break. The eligibility mask removes candidates but **not** truth denominators. To measure end-to-end retrieval, provide all known relevant documents as rows, marking missing candidates ineligible. Otherwise the result is conditional on supplied candidates.

**Algorithm boundary:** the code uses LambdaRank-style lambdas to update a neural network. **It does not implement LambdaMART**, which fits boosted regression trees to ranking gradients (with implementation-specific curvature/leaf updates). No LightGBM/XGBoost adapter is shipped or silently required. The lesson and assignments explain this extension; do not label notebook outputs “LambdaMART.” No cross-encoder/text model is downloaded; features are synthetic numeric query-document features.

## Experiment API and contracts

- `randomize(unique_string_unit_ids, treatment_probability=.5, seed=23)`: deterministic SHA256-derived Bernoulli assignments, order-independent and stable for a unit. This is not exact fixed-size allocation.
- `estimate_ate(outcomes, assignment, confidence=.95)`: difference in means, `sqrt(s1²/n1+s0²/n0)`, normal interval and arm counts. Minimum two independent units per arm; finite outcomes and binary assignment required.
- `srm_pvalue(n_treated, n_control, treatment_probability=.5)`: chi-square allocation test with one degree of freedom; expected counts must be at least five.

The interval is **asymptotic**, not exact small-sample, sequential, clustered or multiple-testing corrected. Two observations per arm make arithmetic possible, not trustworthy asymptotics. Causal interpretation requires randomized assignment, consistency, no interference and no selective missingness. Aggregate repeated events to independent randomization units before inference. SRM detects a count inconsistency, not treatment quality. A valid 95% CI can miss the planted effect on one random realization; tests separately check analytical formulas and repeated-simulation coverage.

## Forecast API and contracts

- `training_windows(prefix, lookback, horizon)`: only complete input/target windows contained within a known prefix.
- `rolling_origins(length, initial, horizon, step)`: expanding-window cutoff indices; target starts exactly at `origin`.
- `seasonal_naive(history, horizon, season)`: repeats the last fully observed season, including horizons longer than a season.
- `TinyNBeats(lookback, horizon, blocks=2)`: MLP backcast/forecast blocks, subtract backcasts from residual input, add block forecasts.
- `train_forecaster(prefix, ...)`: fit mean/std and model on prefix only; returns model, loss history and transform parameters.
- `rolling_forecast(series, ...)`: refit independently at each origin; returns predictions, baseline, targets, MAE, scaling parameters and losses. Mutating future values cannot change predictions at an earlier origin.

**Algorithm boundary:** this is a small generic **N-BEATS-like** architecture, not the full N-BEATS paper, interpretable trend/seasonality stacks or an ensemble. The final backcast is unused by forecast loss because there is no downstream block. **TFT and Prophet are conceptual comparisons only**, not aliases for this network or false full implementations. No prediction interval is fabricated from point forecasts; overlapping rolling errors are dependent.

See `SOLUTION_NOTES.md`, `COMPUTE.md`, and the three assignments for derivations, failure cases and extensions.
