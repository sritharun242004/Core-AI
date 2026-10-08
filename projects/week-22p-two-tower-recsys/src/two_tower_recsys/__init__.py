"""Track P alpha: explicit retrieval, split, metric, and graph contracts."""

from .data import Interaction, load_movielens, synthetic_movielens, temporal_split
from .metrics import RankingMetrics, ranking_metrics
from .model import (
    Popularity,
    RetrievalMetrics,
    TwoTower,
    evaluate,
    recommend,
    train,
    training_pairs,
)

__all__ = [
    "Interaction",
    "Popularity",
    "RankingMetrics",
    "RetrievalMetrics",
    "TwoTower",
    "evaluate",
    "load_movielens",
    "ranking_metrics",
    "recommend",
    "synthetic_movielens",
    "temporal_split",
    "train",
    "training_pairs",
]
