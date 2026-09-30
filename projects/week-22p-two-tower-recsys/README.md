# Track P α — Two-tower recommendation

A real, tiny PyTorch embedding model learns implicit preferences; a one-hop LightGCN-style variant propagates **training edges only**. Compare both with a popularity baseline on chronologically held-out user/item interactions. The default 24-user/18-item fixture is synthetic MovieLens-shaped data, **not MovieLens or a benchmark reproduction**.

## Run offline (Python 3.13)

From the repository root, using the existing environment:

```bash
ROOT="$PWD"
cd projects/week-22p-two-tower-recsys
PYTHONPATH="$PWD/src" "$ROOT/.venv/bin/python" -m pytest
PYTHONPATH="$PWD/src" "$ROOT/.venv/bin/python" notebooks/01-retrieval.py
"$ROOT/.venv/bin/ruff" check src tests notebooks
```

For standalone installation, install this package and its `dev` extra in your own Python 3.13 environment. There are no downloads, API calls, GPU requirements, or cross-project imports. The percent-format notebook is both an ordinary Python script and a Jupytext-compatible notebook source.

## Public API

- `Interaction(user, item, timestamp)`: nonnegative integer IDs/times.
- `synthetic_movielens(seed=22)`: 120 positive events in three planted taste groups.
- `load_movielens(path, min_rating=4)`: optional **local** modern `ratings.csv` parser; returns interactions and raw-ID maps. Does not download data; legacy `.dat` formats are not supported.
- `temporal_split(rows, holdout=1)`: last events per user, disjoint unique edges, strict time boundary. This is per-user chronology, not a global production cutoff.
- `TwoTower(n_users, n_items, dim=12, graph_train=None)`: independently trainable user/item embeddings; inner-product score. Set `graph_train=fit` for one-hop normalized bipartite aggregation averaged with the original embeddings.
- `train(model, fit, steps=100)`: Adam on mean BPR softplus plus L2; returns pre-update losses and final loss. Exhaustive unobserved negatives are chosen from **training information only**.
- `recommend(model, user, fit, k=10)`: exclude seen items, sort ties by ascending item ID, return up to k eligible IDs. Unknown/untrained user IDs fall back to training popularity.
- `Popularity(fit, n_items)`: same seen-item/tie contracts.
- `ranking_metrics(ranked, relevance, k=10)`: full-relevance Recall and graded NDCG; duplicate predictions/invalid grades rejected, empty relevance returns zero. Relevant items missing from retrieval still count in the denominators.
- `evaluate(model, fit, held, k=10)`: macro mean over heldout users; returns Recall, NDCG, and denominator `users`. Disallows overlapping train/test edges.

## Evaluation and scope

The notebook trains on 96 events and evaluates 24 heldout positives, without tuning on their scores. Compare the same full catalog and k across models. Synthetic quality shows that the planted structure is learnable; it does not establish generalization on real preferences. Report heldout counts, cutoff, exclusion policy and popularity results alongside any learned score.

This is an ID-only collaborative model, not a content-feature cold-start tower. Cold-start users receive popularity; unseen items have no useful learned embedding. Real content towers can encode metadata but require independent validation. Graph propagation is a **LightGCN-style one-layer variant**, not a full GCN/GAT, multi-layer reproduction, fraud detector, or molecular model. Dense adjacency and exhaustive negatives scale poorly. There is no ANN index, auction, logged-propensity estimator, calibrated CTR prediction, or online experiment. The next Track P lesson covers ranking, A/B uncertainty and forecasting.
