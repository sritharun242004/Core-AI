import math

import pytest
import torch
from learning_to_rank_timeseries import (
    NeuralRanker,
    group_split,
    grouped_metrics,
    lambda_gradients,
    pairwise_loss,
    ranking_fixture,
    train_ranker,
)

torch.set_num_threads(1)


def test_pairwise_loss_gradient_and_large_margins():
    scores = torch.tensor([0.0, 0.0], requires_grad=True)
    labels, groups = torch.tensor([1.0, 0.0]), torch.tensor([0, 0])
    loss = pairwise_loss(scores, labels, groups)
    assert float(loss.detach()) == pytest.approx(math.log(2))
    loss.backward()
    assert scores.grad.tolist() == pytest.approx([-0.5, 0.5])
    assert torch.isfinite(pairwise_loss(torch.tensor([-1000.0, 1000.0]), labels, groups))


def test_lambda_weights_are_exact_ndcg_swap_change_not_plain_ranknet():
    scores = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    labels, groups = torch.tensor([1.0, 0.0]), torch.tensor([0, 0])
    delta = 1 - 1 / math.log2(3)
    expected = torch.tensor([-0.5 * delta, 0.5 * delta], dtype=torch.float64)
    assert torch.allclose(lambda_gradients(scores, labels, groups, k=2), expected)
    pairwise_loss(scores, labels, groups, method="lambda", k=2).backward()
    assert torch.allclose(scores.grad, expected)


def test_lambda_gradient_finite_difference_away_from_ties():
    scores = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    labels, groups = torch.tensor([2.0, 0.0, 1.0]), torch.zeros(3, dtype=torch.long)
    analytic = lambda_gradients(scores, labels, groups, k=2)
    for j in range(3):
        direction = torch.zeros(3, dtype=torch.float64)
        direction[j] = 1e-6
        high = pairwise_loss(scores + direction, labels, groups, method="lambda", k=2)
        low = pairwise_loss(scores - direction, labels, groups, method="lambda", k=2)
        assert analytic[j] == pytest.approx(float((high - low) / 2e-6), abs=1e-7)


def test_no_cross_query_pairs_and_tied_labels_have_zero_gradient():
    scores = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    labels, groups = torch.tensor([2.0, 2.0, 0.0]), torch.tensor([1, 1, 2])
    loss = pairwise_loss(scores, labels, groups)
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(scores.grad, torch.zeros(3))


def test_macro_metrics_empty_relevance_stable_ties_and_masks():
    scores = torch.zeros(4)
    labels, groups = torch.tensor([0.0, 1.0, 0.0, 0.0]), torch.tensor([0, 0, 1, 1])
    report = grouped_metrics(scores, labels, groups, k=1)
    assert report == {"ndcg": 0.0, "recall": 0.0, "queries": 2}
    # Masked high-relevance document still counts in the relevance denominator.
    mask = torch.tensor([True, False, True, True])
    assert grouped_metrics(scores, labels, groups, k=2, eligible=mask)["recall"] == 0
    assert grouped_metrics(scores, labels, groups, k=2)["recall"] == 0.5


def test_macro_loss_includes_zero_pair_groups_and_lambda_normalization():
    scores = torch.zeros(4, requires_grad=True)
    relevance, groups = torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([0, 0, 1, 1])
    assert pairwise_loss(scores, relevance, groups).item() == pytest.approx(math.log(2) / 2)
    loss = pairwise_loss(scores, relevance, groups, method="lambda", k=2)
    loss.backward()
    assert torch.allclose(scores.grad, lambda_gradients(scores, relevance, groups, k=2))
    assert scores.grad[2:].tolist() == [0.0, 0.0]


def test_all_masked_queries_and_empty_inputs_have_explicit_denominators():
    scores, relevance, groups = torch.zeros(2), torch.tensor([2.0, 1.0]), torch.tensor([0, 0])
    report = grouped_metrics(scores, relevance, groups, eligible=torch.zeros(2, dtype=torch.bool))
    assert report == {"ndcg": 0.0, "recall": 0.0, "queries": 1}
    empty = torch.empty(0)
    assert grouped_metrics(empty, empty, torch.empty(0, dtype=torch.long)) == {
        "ndcg": 0.0,
        "recall": 0.0,
        "queries": 0,
    }


def test_group_split_is_query_disjoint_and_deterministic():
    _, _, groups = ranking_fixture()
    fit, held = group_split(groups)
    assert set(groups[fit].tolist()).isdisjoint(groups[held].tolist())
    assert torch.equal(fit, group_split(groups)[0])
    assert len(fit) + len(held) == len(groups)


@pytest.mark.parametrize("method", ["pairwise", "lambda"])
def test_neural_reranker_actually_learns_and_generalizes(method):
    x, labels, groups = ranking_fixture()
    fit, held = group_split(groups)
    model = NeuralRanker(x.shape[1])
    before = [p.detach().clone() for p in model.parameters()]
    history = train_ranker(model, x[fit], labels[fit], groups[fit], method=method, steps=80)
    assert history[-1] < history[0] * 0.2
    assert all(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    assert grouped_metrics(model(x[held]), labels[held], groups[held], k=3)["ndcg"] > 0.9


@pytest.mark.parametrize("bad", [float("nan"), -1.0])
def test_invalid_relevance_rejected(bad):
    with pytest.raises(ValueError):
        pairwise_loss(torch.zeros(2), torch.tensor([1.0, bad]), torch.zeros(2, dtype=torch.long))
