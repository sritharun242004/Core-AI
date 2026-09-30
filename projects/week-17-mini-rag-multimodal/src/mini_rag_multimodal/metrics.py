"""Ranking metrics with explicit edge-case contracts."""

import math
from collections.abc import Mapping, Sequence


def _validate(ranking: Sequence[str], relevance: Mapping[str, float], k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking contains a duplicate document id")
    for value in relevance.values():
        if value < 0:
            raise ValueError("relevance must be nonnegative")
        if not math.isfinite(value):
            raise ValueError("relevance must be finite")


def _positive(relevance: Mapping[str, float]) -> set[str]:
    return {doc_id for doc_id, value in relevance.items() if value > 0}


def recall_at_k(ranking: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    _validate(ranking, relevance, k)
    positives = _positive(relevance)
    return len(set(ranking[:k]) & positives) / len(positives) if positives else 0.0


def reciprocal_rank_at_k(ranking: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    _validate(ranking, relevance, k)
    positives = _positive(relevance)
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if doc_id in positives:
            return 1 / rank
    return 0.0


def average_precision_at_k(ranking: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    _validate(ranking, relevance, k)
    positives = _positive(relevance)
    if not positives:
        return 0.0
    precision_sum = 0.0
    found = 0
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if doc_id in positives:
            found += 1
            precision_sum += found / rank
    return precision_sum / len(positives)


def ndcg_at_k(ranking: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    _validate(ranking, relevance, k)
    if not _positive(relevance):
        return 0.0

    def gain(value: float) -> float:
        return 2**value - 1

    actual = sum(
        gain(relevance.get(doc_id, 0.0)) / math.log2(rank)
        for rank, doc_id in enumerate(ranking[:k], start=2)
    )
    ideal_values = sorted(relevance.values(), reverse=True)[:k]
    ideal = sum(gain(value) / math.log2(rank) for rank, value in enumerate(ideal_values, start=2))
    return actual / ideal if ideal else 0.0


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, float]], k: int = 5
) -> dict[str, float]:
    """Macro-average queries, including missing rankings as zero.

    Including qrels with no result is deliberate: an evaluator must not silently
    remove failed queries from the denominator.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    query_ids = set(qrels) | set(rankings)
    if not query_ids:
        return {f"recall@{k}": 0.0, f"mrr@{k}": 0.0, f"ndcg@{k}": 0.0, f"map@{k}": 0.0}
    values = [
        (
            recall_at_k(rankings.get(query_id, []), qrels.get(query_id, {}), k),
            reciprocal_rank_at_k(rankings.get(query_id, []), qrels.get(query_id, {}), k),
            ndcg_at_k(rankings.get(query_id, []), qrels.get(query_id, {}), k),
            average_precision_at_k(rankings.get(query_id, []), qrels.get(query_id, {}), k),
        )
        for query_id in sorted(query_ids)
    ]
    mean = [sum(row[index] for row in values) / len(values) for index in range(4)]
    return {
        f"recall@{k}": mean[0],
        f"mrr@{k}": mean[1],
        f"ndcg@{k}": mean[2],
        f"map@{k}": mean[3],
    }
