# pyright: reportMissingParameterType=false, reportUnknownParameterType=false
import math

import pytest
from mini_rag_multimodal.metrics import (
    average_precision_at_k,
    evaluate_rankings,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_recall_reciprocal_rank_and_ap_have_hand_computable_values() -> None:
    ranking = ["irrelevant", "a", "b"]
    labels = {"a": 1, "b": 1, "unretrieved": 1}
    assert recall_at_k(ranking, labels, 2) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k(ranking, labels, 3) == 0.5
    # AP@k divides by ALL relevant documents, not just the retrieved positives.
    assert average_precision_at_k(ranking, labels, 3) == pytest.approx((0.5 + 2 / 3) / 3)


def test_graded_ndcg_uses_exponential_gain_and_ideal_order() -> None:
    labels = {"a": 3, "b": 1}
    expected = (1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], labels, 2) == pytest.approx(expected)
    assert ndcg_at_k(["a", "b"], labels, 2) == pytest.approx(1)


def test_macro_metrics_count_missing_queries_as_zero() -> None:
    report = evaluate_rankings({"q1": ["x", "a"]}, {"q1": {"a": 1}, "q2": {"b": 1}}, k=2)
    assert report["recall@2"] == 0.5
    assert report["mrr@2"] == 0.25
    assert report["map@2"] == 0.25
    assert report["ndcg@2"] == pytest.approx(0.5 / math.log2(3))


@pytest.mark.parametrize(
    "metric", [recall_at_k, reciprocal_rank_at_k, average_precision_at_k, ndcg_at_k]
)
def test_empty_relevance_and_empty_results_are_zero(metric) -> None:
    assert metric([], {"a": 1}, 3) == 0.0
    assert metric(["a"], {}, 3) == 0.0
    assert metric(["a"], {"a": 0}, 3) == 0.0


@pytest.mark.parametrize(
    "metric", [recall_at_k, reciprocal_rank_at_k, average_precision_at_k, ndcg_at_k]
)
def test_invalid_metric_inputs_are_rejected(metric) -> None:
    with pytest.raises(ValueError, match="positive"):
        metric(["a"], {"a": 1}, 0)
    with pytest.raises(ValueError, match="duplicate"):
        metric(["a", "a"], {"a": 1}, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        metric(["a"], {"a": -1}, 2)
    with pytest.raises(ValueError, match="finite"):
        metric(["a"], {"a": float("nan")}, 2)


def test_no_labeled_queries_is_a_zero_report() -> None:
    assert evaluate_rankings({}, {}, k=2) == {
        "recall@2": 0.0,
        "mrr@2": 0.0,
        "ndcg@2": 0.0,
        "map@2": 0.0,
    }
