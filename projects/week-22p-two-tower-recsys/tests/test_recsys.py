# pyright: reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
import math

import pytest
import torch
from two_tower_recsys import (
    Interaction,
    Popularity,
    TwoTower,
    evaluate,
    load_movielens,
    ranking_metrics,
    recommend,
    synthetic_movielens,
    temporal_split,
    train,
    training_pairs,
)

torch.set_num_threads(1)


def test_temporal_holdout_has_no_shared_edges_and_preserves_time():
    rows = synthetic_movielens()
    fit, held = temporal_split(rows)
    assert {(r.user, r.item) for r in fit}.isdisjoint((r.user, r.item) for r in held)
    for user in {r.user for r in rows}:
        assert max(r.timestamp for r in fit if r.user == user) < min(
            r.timestamp for r in held if r.user == user
        )
    assert len(held) == 24


def test_duplicate_edges_and_tied_split_boundary_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        temporal_split([Interaction(0, 0, 1), Interaction(0, 0, 2)])
    with pytest.raises(ValueError, match="timestamp"):
        temporal_split([Interaction(0, 0, 1), Interaction(0, 1, 1)])
    with pytest.raises(ValueError):
        temporal_split([Interaction(0, 0, 1)])


def test_recall_denominator_is_all_relevance_not_retrieved_relevance():
    result = ranking_metrics([9, 2], {2: 1, 3: 1, 4: 1}, k=2)
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["ndcg"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert ranking_metrics([], {}, k=3) == {"recall": 0.0, "ndcg": 0.0}
    assert ranking_metrics([1], {1: 0}, k=1)["ndcg"] == 0


@pytest.mark.parametrize(
    "ranking,relevance,k",
    [([1, 1], {1: 1}, 2), ([1], {1: -1}, 1), ([1], {1: float("nan")}, 1), ([1], {1: 1}, 0)],
)
def test_metric_invalid_contract(ranking, relevance, k):
    with pytest.raises(ValueError):
        ranking_metrics(ranking, relevance, k=k)


def test_graded_ndcg_and_full_catalog_ideal():
    result = ranking_metrics([1, 2], {1: 1, 2: 3, 7: 2}, k=2)
    assert result["ndcg"] == pytest.approx((1 + 7 / math.log2(3)) / (7 + 3 / math.log2(3)))


def test_pairs_exclude_every_training_positive_and_are_deterministic():
    fit, _ = temporal_split(synthetic_movielens())
    pairs = training_pairs(fit, 18)
    assert all((int(u), int(n)) not in {(r.user, r.item) for r in fit} for u, _, n in pairs)
    assert torch.equal(pairs, training_pairs(fit, 18))
    with pytest.raises(ValueError):
        training_pairs([Interaction(0, 0, 0)], 1)


@pytest.mark.parametrize("graph", [False, True])
def test_real_training_updates_both_towers_and_reduces_loss(graph):
    fit, held = temporal_split(synthetic_movielens())
    model = TwoTower(24, 18, dim=12, graph_train=fit if graph else None)
    before_u = model.user_embedding.weight.detach().clone()
    before_i = model.item_embedding.weight.detach().clone()
    history = train(model, fit, steps=100)
    assert history[-1] < history[0] * 0.15
    assert not torch.equal(before_u, model.user_embedding.weight)
    assert not torch.equal(before_i, model.item_embedding.weight)
    assert model.user_embedding.weight.grad.abs().sum() > 0
    assert model.item_embedding.weight.grad.abs().sum() > 0
    report = evaluate(model, fit, held, k=4)
    assert report["users"] == 24
    assert 0 <= report["recall"] <= 1 and 0 <= report["ndcg"] <= 1
    assert report["recall"] > 0.5  # Planted groups, not a MovieLens benchmark.


def test_graph_holdout_edges_absent_in_both_directions():
    fit, held = temporal_split(synthetic_movielens())
    model = TwoTower(24, 18, graph_train=fit)
    for row in held:
        assert model.adjacency[row.user, 24 + row.item] == 0
        assert model.adjacency[24 + row.item, row.user] == 0
    assert torch.equal(model.adjacency, model.adjacency.T)


def test_bpr_score_gradients_match_finite_differences():
    model = TwoTower(1, 2, dim=2).double()
    user, positive, negative = torch.tensor([0]), torch.tensor([0]), torch.tensor([1])

    def loss():
        return torch.nn.functional.softplus(-(model(user, positive) - model(user, negative))).mean()

    loss().backward()
    for parameter in (model.user_embedding.weight, model.item_embedding.weight):
        expected = float(parameter.grad[0, 0])
        with torch.no_grad():
            original = parameter[0, 0].item()
            parameter[0, 0] = original + 1e-6
            high = loss().item()
            parameter[0, 0] = original - 1e-6
            low = loss().item()
            parameter[0, 0] = original
        assert expected == pytest.approx((high - low) / 2e-6, abs=1e-8)


def test_graph_contains_only_training_edges_and_normalized_neighbors():
    fit = [Interaction(0, 0, 0), Interaction(0, 1, 1), Interaction(1, 1, 0)]
    model = TwoTower(2, 3, dim=2, graph_train=fit)
    a = model.adjacency
    assert a[0, 2] == pytest.approx(1 / math.sqrt(2))
    assert a[0, 3] == pytest.approx(0.5)
    assert a[1, 2] == 0 and a[:, 4].sum() == 0
    raw = torch.cat([model.user_embedding.weight, model.item_embedding.weight])
    u, i = model.embeddings()
    assert torch.allclose(torch.cat([u, i]), (raw + a @ raw) / 2)
    with pytest.raises(ValueError, match="graph"):
        train(model, fit[:2], steps=1)


def test_recommendation_masks_seen_stably_breaks_ties_and_handles_all_seen():
    fit = [Interaction(0, 1, 0)]
    model = TwoTower(2, 4)
    with torch.no_grad():
        model.user_embedding.weight.zero_()
    assert recommend(model, 0, fit, k=8) == [0, 2, 3]
    assert recommend(model, 0, [Interaction(0, i, i) for i in range(4)], k=4) == []
    with pytest.raises(ValueError):
        recommend(model, 0, fit, k=0)


def test_popularity_cold_start_never_indexes_unknown_user():
    fit = [Interaction(0, 2, 0), Interaction(1, 2, 0), Interaction(1, 1, 1)]
    model = TwoTower(3, 4)
    assert Popularity(fit, 4).recommend(0, k=4) == [1, 0, 3]
    assert recommend(model, 999, fit, k=2) == [2, 1]
    assert recommend(model, 2, fit, k=2) == [2, 1]  # In-range but untrained ID.


def test_evaluation_cannot_include_seen_holdout_and_handles_empty():
    fit = [Interaction(0, 0, 0)]
    model = TwoTower(1, 3)
    with pytest.raises(ValueError, match="overlap"):
        evaluate(model, fit, fit)
    assert evaluate(model, fit, [], k=2) == {"recall": 0.0, "ndcg": 0.0, "users": 0}
    assert evaluate(Popularity(fit, 3), fit, [Interaction(0, 1, 1)], k=1)["recall"] == 1


def test_optional_local_parser_maps_ids_and_filters_ratings(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text("userId,movieId,rating,timestamp\n10,99,5,1\n10,20,4,2\n11,99,2,3\n")
    rows, users, items = load_movielens(path)
    assert users == {10: 0} and items == {20: 0, 99: 1}
    assert rows == [Interaction(0, 1, 1), Interaction(0, 0, 2)]
    assert not (tmp_path / "download").exists()
