"""Tiny ID towers and one-hop LightGCN-style train-only propagation.

Dense graph and exhaustive negatives are deliberately pedagogical, not scalable.
"""

from collections import defaultdict
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import cast

import torch
from torch import nn
from torch.nn import functional

from .data import Interaction
from .metrics import RankingMetrics, positive_integer, ranking_metrics


class RetrievalMetrics(RankingMetrics):
    users: int


def _validate_rows(rows: list[Interaction], n_users: int, n_items: int) -> None:
    if any(r.user >= n_users or r.item >= n_items for r in rows):
        raise ValueError("interaction ID outside configured vocabulary")
    if len({(r.user, r.item) for r in rows}) != len(rows):
        raise ValueError("duplicate training edges")


class TwoTower(nn.Module):
    """Separate trainable user and item embedding towers; score is inner product."""

    adjacency: torch.Tensor

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 12,
        *,
        seed: int = 22,
        graph_train: list[Interaction] | None = None,
    ) -> None:
        super().__init__()
        if min(n_users, n_items, dim) < 1:
            raise ValueError("positive embedding dimensions required")
        self.n_users, self.n_items = n_users, n_items
        # PyTorch leaves these callable signatures partially unannotated.
        with cast(Callable[[], AbstractContextManager[None]], torch.random.fork_rng)():
            cast(Callable[[int], torch.Generator], torch.manual_seed)(seed)
            self.user_embedding = nn.Embedding(n_users, dim)
            self.item_embedding = nn.Embedding(n_items, dim)
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)
        self.graph_edges: frozenset[tuple[int, int]] | None = None
        adjacency = torch.empty(0)
        if graph_train is not None:
            _validate_rows(graph_train, n_users, n_items)
            self.graph_edges = frozenset((r.user, r.item) for r in graph_train)
            adjacency = torch.zeros(n_users + n_items, n_users + n_items)
            for user, item in self.graph_edges:
                adjacency[user, n_users + item] = adjacency[n_users + item, user] = 1
            inv_degree = adjacency.sum(1).clamp_min(1).rsqrt()
            adjacency = inv_degree[:, None] * adjacency * inv_degree[None, :]
        self.register_buffer("adjacency", adjacency)

    def embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.graph_edges is None:
            return self.user_embedding.weight, self.item_embedding.weight
        raw = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        propagated = (raw + self.adjacency @ raw) / 2
        return propagated[: self.n_users], propagated[self.n_users :]

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u, i = self.embeddings()
        return (u[users] * i[items]).sum(-1)


def training_pairs(rows: list[Interaction], n_items: int) -> torch.Tensor:
    """All (user, observed-positive, unobserved-item) triples, training data only.

    Unobserved is not a known dislike. Held-out labels are deliberately unavailable
    to this sampler; false negatives remain possible in implicit-feedback data.
    """
    _validate_rows(rows, max((r.user for r in rows), default=0) + 1, n_items)
    seen: defaultdict[int, set[int]] = defaultdict(set)
    for row in rows:
        seen[row.user].add(row.item)
    triples = [
        (r.user, r.item, item) for r in rows for item in range(n_items) if item not in seen[r.user]
    ]
    if not triples:
        raise ValueError("training requires a positive and an unobserved item")
    return torch.tensor(triples, dtype=torch.long)


def train(
    model: TwoTower,
    rows: list[Interaction],
    *,
    steps: int = 100,
    lr: float = 0.03,
    l2: float = 1e-4,
) -> list[float]:
    if steps < 1 or lr <= 0 or l2 < 0:
        raise ValueError("steps/lr must be positive; l2 nonnegative")
    _validate_rows(rows, model.n_users, model.n_items)
    if model.graph_edges is not None and model.graph_edges != frozenset(
        (r.user, r.item) for r in rows
    ):
        raise ValueError("graph edges must exactly match training positives")
    triples = training_pairs(rows, model.n_items).to(model.user_embedding.weight.device)
    user, positive, negative = triples.unbind(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    model.train()
    for step in range(steps + 1):
        u, i = model.embeddings()
        margin = (u[user] * (i[positive] - i[negative])).sum(-1)
        loss = functional.softplus(-margin).mean() + l2 * sum(
            p.square().mean() for p in model.parameters()
        )
        history.append(float(loss.detach()))
        if step == steps:
            break
        optimizer.zero_grad()
        cast(Callable[[], None], loss.backward)()
        cast(Callable[[], None], optimizer.step)()
    return history


class Popularity:
    """Training counts only; stable item-ID tie break and seen-item exclusion."""

    def __init__(self, rows: list[Interaction], n_items: int) -> None:
        if n_items < 1:
            raise ValueError("catalog must not be empty")
        _validate_rows(rows, max((r.user for r in rows), default=0) + 1, n_items)
        self.n_items = n_items
        self.counts = [0] * n_items
        self.seen: defaultdict[int, set[int]] = defaultdict(set)
        for row in rows:
            self.counts[row.item] += 1
            self.seen[row.user].add(row.item)

    def recommend(self, user: int, *, k: int = 10) -> list[int]:
        if not positive_integer(k):
            raise ValueError("k must be positive")
        eligible = (i for i in range(self.n_items) if i not in self.seen.get(user, set()))
        return sorted(eligible, key=lambda i: (-self.counts[i], i))[:k]


@torch.no_grad()
def recommend(model: TwoTower, user: int, rows: list[Interaction], *, k: int = 10) -> list[int]:
    if not positive_integer(k):
        raise ValueError("k must be positive")
    baseline = Popularity(rows, model.n_items)
    if not 0 <= user < model.n_users or user not in baseline.seen:
        return baseline.recommend(user, k=k)
    u, i = model.embeddings()
    # A matrix-vector product is a 1-D floating tensor, so tolist yields scalar scores.
    scores = cast(list[float], (i @ u[user]).cpu().tolist())
    if not all(torch.isfinite(torch.tensor(scores))):
        raise ValueError("nonfinite retrieval scores")
    eligible = (item for item in range(model.n_items) if item not in baseline.seen[user])
    return sorted(eligible, key=lambda item: (-scores[item], item))[:k]


def evaluate(
    model: TwoTower | Popularity, fit: list[Interaction], held: list[Interaction], *, k: int = 10
) -> RetrievalMetrics:
    if k < 1:
        raise ValueError("k must be positive")
    if {(r.user, r.item) for r in fit} & {(r.user, r.item) for r in held}:
        raise ValueError("training and heldout edges overlap")
    if any(r.item >= model.n_items for r in held):
        raise ValueError("heldout item is outside candidate catalog")
    relevant: defaultdict[int, dict[int, float]] = defaultdict(dict)
    for row in held:
        relevant[row.user][row.item] = 1.0
    results: list[RankingMetrics] = []
    for user, labels in relevant.items():
        ranked = (
            model.recommend(user, k=k)
            if isinstance(model, Popularity)
            else recommend(model, user, fit, k=k)
        )
        results.append(ranking_metrics(ranked, labels, k=k))
    return {
        "recall": sum(r["recall"] for r in results) / len(results) if results else 0.0,
        "ndcg": sum(r["ndcg"] for r in results) / len(results) if results else 0.0,
        "users": len(results),
    }
