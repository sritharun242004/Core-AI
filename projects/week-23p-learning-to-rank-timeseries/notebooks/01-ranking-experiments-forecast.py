# %% [markdown]
# # Track P beta: separate offline quality, causal evidence and forecasting
# Actual neural optimization on synthetic data; no external models or datasets.

# %%
import numpy as np
import torch
from learning_to_rank_timeseries import (
    NeuralRanker,
    estimate_ate,
    forecast_fixture,
    group_split,
    grouped_metrics,
    randomize,
    ranking_fixture,
    rolling_forecast,
    srm_pvalue,
    train_ranker,
)

torch.set_num_threads(1)
x, relevance, groups = ranking_fixture()
fit, held = group_split(groups)
assert set(groups[fit].tolist()).isdisjoint(groups[held].tolist())
for method in ("pairwise", "lambda"):
    model = NeuralRanker(x.shape[1])
    history = train_ranker(model, x[fit], relevance[fit], groups[fit], method=method)
    assert history[-1] < history[0]
    print(method, "loss", history[0], history[-1])
    print("heldout", grouped_metrics(model(x[held]), relevance[held], groups[held], k=3))

# %% [markdown]
# The lambda model is a neural frozen-rank LambdaRank-style surrogate, NOT
# LambdaMART/boosted trees. Query metrics are conditional on supplied truth rows.

# %%
rng = np.random.default_rng(23)
assignment = randomize([str(i) for i in range(10000)])
outcomes = rng.normal(size=10000) + 0.4 * assignment
report = estimate_ate(outcomes, assignment)
print("experiment", report)
print("SRM p-value", srm_pvalue(report.n_treated, report.n_control))
print("planted effect 0.4 covered by this 95% interval?", report.low <= 0.4 <= report.high)
# A 95% confidence procedure can miss the planted effect in an individual run.
# Do not select another seed simply to obtain a more pleasing interval.

# %%
series = forecast_fixture(84)
reports = rolling_forecast(series, initial=60, horizon=4, step=8, steps=100)
for row in reports:
    print("origin", row["origin"], "neural MAE", row["mae"], "seasonal MAE", row["baseline_mae"])
    assert row["final_loss"] < row["initial_loss"]
changed = series.copy()
changed[60:] += 1000
counterfactual = rolling_forecast(changed, initial=60, horizon=4, step=24, steps=100)
assert np.array_equal(reports[0]["forecast"], counterfactual[0]["forecast"])
print("future-mutation invariance: passed")

# %% [markdown]
# The neural architecture has genuine residual backcast/forecast blocks, but is
# N-BEATS-like, not a full paper reproduction, TFT or Prophet. No forecasting
# interval is claimed. Rolling origins can have dependent errors. A/B confidence
# needs independent randomized units and stated causal assumptions.
