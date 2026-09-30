# %% [markdown]
# # Track P alpha: actual tiny retrieval training, honest heldout evaluation
# Synthetic MovieLens-shaped interactions, not downloaded MovieLens records.
# The same full catalog and seen-item exclusion apply to every model.

# %%
import torch
from two_tower_recsys import (
    Popularity,
    TwoTower,
    evaluate,
    ranking_metrics,
    recommend,
    synthetic_movielens,
    temporal_split,
    train,
)

torch.set_num_threads(1)
fit, held = temporal_split(synthetic_movielens())
print(f"train={len(fit)}, heldout={len(held)}, users=24, catalog=18")
print("popularity", evaluate(Popularity(fit, 18), fit, held, k=4))

# %%
for graph in (False, True):
    model = TwoTower(24, 18, graph_train=fit if graph else None)
    if graph:
        assert all(model.adjacency[r.user, 24 + r.item] == 0 for r in held)
    losses = train(model, fit, steps=100)
    assert losses[-1] < losses[0]
    print("graph" if graph else "ID towers", {"initial_loss": losses[0], "final_loss": losses[-1]})
    print("heldout", evaluate(model, fit, held, k=4))
    print("user 0", recommend(model, 0, fit, k=4))
    print("cold-start fallback", recommend(model, 999, fit, k=4))

# %%
print("denominator example", ranking_metrics([9, 2], {2: 1, 3: 1, 4: 1}, k=2))
print("empty relevance", ranking_metrics([], {}, k=2))

# %% [markdown]
# The split is chronological per user, not a global launch-time evaluation.
# No hyperparameters were selected with heldout labels. This planted-data result
# cannot establish real MovieLens quality, causal lift, cold-item generalization,
# fairness or scalable ANN throughput. See README and SOLUTION_NOTES.
