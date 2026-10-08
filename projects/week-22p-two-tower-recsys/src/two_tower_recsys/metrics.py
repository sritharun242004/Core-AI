"""Full-relevance denominators; empty relevance contributes zero to macro means."""

import math
from collections.abc import Mapping, Sequence
from typing import TypedDict


class RankingMetrics(TypedDict):
    recall: float
    ndcg: float


def positive_integer(value: object) -> bool:
    """Runtime check for callers whose inputs have not been statically checked."""
    return isinstance(value, int) and value >= 1


def ranking_metrics(
    ranked: Sequence[int], relevance: Mapping[int, float], *, k: int = 10
) -> RankingMetrics:
    if not positive_integer(k):
        raise ValueError("k must be a positive integer")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked item IDs must be unique")
    if any(not math.isfinite(v) or v < 0 or v > 50 for v in relevance.values()):
        raise ValueError("relevance must be finite in [0, 50]")
    positives = {item for item, grade in relevance.items() if grade > 0}
    top = ranked[:k]
    recall = len(set(top) & positives) / len(positives) if positives else 0.0

    def dcg(grades: Sequence[float]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))

    actual = dcg([relevance.get(item, 0.0) for item in top])
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return {"recall": recall, "ndcg": actual / ideal if ideal else 0.0}
