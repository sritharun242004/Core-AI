"""Grouped RankNet and frozen-rank LambdaRank gradients; NOT LambdaMART."""

import math
from contextlib import AbstractContextManager
from typing import Protocol, TypedDict, cast

import torch
from torch import Tensor, nn
from torch.nn import functional


class GroupedMetrics(TypedDict):
    ndcg: float
    recall: float
    queries: int


class _TorchRandom(Protocol):
    def fork_rng(self) -> AbstractContextManager[None]: ...


class _TorchApi(Protocol):
    random: _TorchRandom

    def unique(self, values: Tensor) -> Tensor: ...

    def manual_seed(self, seed: int) -> torch.Generator: ...


class _TensorValues(Protocol):
    def tolist(self) -> list[float]: ...


class _Backward(Protocol):
    def backward(self) -> None: ...


class _Steppable(Protocol):
    def step(self) -> None: ...


def _torch_unique(values: Tensor) -> Tensor:
    return cast(_TorchApi, torch).unique(values)


def _tensor_values(values: Tensor) -> list[float]:
    return cast(_TensorValues, values).tolist()


def _validate(scores: Tensor, relevance: Tensor, groups: Tensor, k: object) -> None:
    if scores.ndim != 1 or scores.shape != relevance.shape or scores.shape != groups.shape:
        raise ValueError("scores, relevance and groups must be aligned 1-D tensors")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not torch.isfinite(scores).all() or not torch.isfinite(relevance).all():
        raise ValueError("scores and relevance must be finite")
    if ((relevance < 0) | (relevance > 20)).any():
        raise ValueError("relevance grades must be in [0, 20]")
    if groups.dtype not in (torch.int32, torch.int64):
        raise ValueError("query groups must be integer IDs")


def _discounts(size: int, k: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    ranks = torch.arange(size, device=device, dtype=dtype)
    return torch.where(ranks < k, 1 / torch.log2(ranks + 2), 0)


def _swap_weights(scores: Tensor, relevance: Tensor, k: int) -> Tensor:
    """Absolute NDCG@k change from swapping each pair at the CURRENT ranks.

    Stable ties follow input document order; ranks/weights are stop-gradient.
    """
    with torch.no_grad():
        gains = 2.0 ** relevance.to(scores.dtype) - 1
        discounts = _discounts(len(scores), k, device=scores.device, dtype=scores.dtype)
        ideal = (gains.sort(descending=True).values * discounts).sum()
        order = torch.argsort(scores, descending=True, stable=True)
        doc_discounts = torch.empty_like(discounts)
        doc_discounts[order] = discounts
        delta = (gains[:, None] - gains[None, :]).abs() * (
            doc_discounts[:, None] - doc_discounts[None, :]
        ).abs()
        return delta / ideal if ideal > 0 else delta * 0


def pairwise_loss(
    scores: Tensor, relevance: Tensor, groups: Tensor, *, method: str = "pairwise", k: int = 10
) -> Tensor:
    """Mean per-query loss, then macro mean over ALL queries (empty pairs = 0).

    method='lambda' differentiates delta-NDCG weighted logistic losses with ranks
    held fixed: a neural LambdaRank-style update, not differentiable NDCG or trees.
    """
    _validate(scores, relevance, groups, k)
    if method not in ("pairwise", "lambda"):
        raise ValueError("method must be pairwise or lambda")
    losses: list[Tensor] = []
    for group in _torch_unique(groups):
        idx = groups == group
        s, r = scores[idx], relevance[idx]
        preferred = r[:, None] > r[None, :]
        if not preferred.any():
            losses.append(s.sum() * 0)
            continue
        terms = functional.softplus(-(s[:, None] - s[None, :]))
        if method == "lambda":
            terms = terms * _swap_weights(s, r, k)
        losses.append(terms[preferred].mean())
    return torch.stack(losses).mean() if losses else scores.sum() * 0


def lambda_gradients(scores: Tensor, relevance: Tensor, groups: Tensor, *, k: int = 10) -> Tensor:
    """Analytical score gradients of the frozen-rank surrogate, independently coded."""
    _validate(scores, relevance, groups, k)
    gradient = torch.zeros_like(scores)
    unique = _torch_unique(groups)
    with torch.no_grad():
        for group in unique:
            idx = torch.where(groups == group)[0]
            s, r = scores[idx], relevance[idx]
            pairs = torch.nonzero(r[:, None] > r[None, :])
            if not len(pairs):
                continue
            weights = _swap_weights(s, r, k)
            for hi, lo in pairs:
                magnitude = (
                    weights[hi, lo] * torch.sigmoid(s[lo] - s[hi]) / (len(pairs) * len(unique))
                )
                gradient[idx[hi]] -= magnitude
                gradient[idx[lo]] += magnitude
    return gradient


def grouped_metrics(
    scores: Tensor,
    relevance: Tensor,
    groups: Tensor,
    *,
    k: int = 10,
    eligible: Tensor | None = None,
) -> GroupedMetrics:
    """Macro query Recall/NDCG. Mask affects ranking, NEVER truth denominators.

    Truth is complete only within the provided query rows. For end-to-end search,
    supply rows for absent candidates too and mark those rows ineligible.
    """
    _validate(scores, relevance, groups, k)
    if eligible is None:
        eligible = torch.ones_like(groups, dtype=torch.bool)
    if eligible.shape != scores.shape or eligible.dtype != torch.bool:
        raise ValueError("eligible must be an aligned boolean mask")
    metrics: list[tuple[float, float]] = []
    with torch.no_grad():
        for group in _torch_unique(groups):
            idx = torch.where(groups == group)[0]
            candidates = idx[eligible[idx]]
            ranked = candidates[torch.argsort(scores[candidates], descending=True, stable=True)][:k]
            # Validation guarantees 1-D numeric relevance, hence scalar tolist entries.
            grades = _tensor_values(relevance[ranked])
            ideal_grades = sorted(_tensor_values(relevance[idx]), reverse=True)[:k]
            dcg = sum((2**r - 1) / math.log2(t + 2) for t, r in enumerate(grades))
            ideal = sum((2**r - 1) / math.log2(t + 2) for t, r in enumerate(ideal_grades))
            positives = int((relevance[idx] > 0).sum())
            metrics.append(
                (
                    dcg / ideal if ideal else 0.0,
                    sum(r > 0 for r in grades) / positives if positives else 0.0,
                )
            )
    return {
        "ndcg": sum(m[0] for m in metrics) / len(metrics) if metrics else 0.0,
        "recall": sum(m[1] for m in metrics) / len(metrics) if metrics else 0.0,
        "queries": len(metrics),
    }


def ranking_fixture(
    seed: int = 23, queries: int = 30, documents: int = 6
) -> tuple[Tensor, Tensor, Tensor]:
    if queries < 2 or documents < 2:
        raise ValueError("fixture requires at least two queries and documents")
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(queries, documents, 3, generator=generator)
    utility = 2 * x[:, :, 0] - x[:, :, 1] + 0.4 * x[:, :, 2]
    order = utility.argsort(dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(documents).expand(queries, -1))
    relevance = (ranks * 3 // documents).float()
    groups = torch.arange(queries).repeat_interleave(documents)
    return x.reshape(-1, 3), relevance.reshape(-1), groups


def group_split(
    groups: Tensor, *, test_fraction: float = 0.3, seed: int = 23
) -> tuple[Tensor, Tensor]:
    if groups.ndim != 1 or groups.dtype not in (torch.int32, torch.int64):
        raise ValueError("groups must be a one-dimensional integer tensor")
    unique = _torch_unique(groups)
    if len(unique) < 2 or not 0 < test_fraction < 1:
        raise ValueError("need multiple query groups and a fraction in (0,1)")
    perm = torch.randperm(len(unique), generator=torch.Generator().manual_seed(seed))
    n_test = max(1, min(len(unique) - 1, round(len(unique) * test_fraction)))
    held = torch.isin(groups, unique[perm[:n_test]])
    return torch.where(~held)[0], torch.where(held)[0]


class NeuralRanker(nn.Module):
    def __init__(self, features: int, hidden: int = 16, seed: int = 23) -> None:
        super().__init__()
        if min(features, hidden) < 1:
            raise ValueError("positive network dimensions required")
        # PyTorch leaves these callable signatures partially unannotated.
        torch_api = cast(_TorchApi, torch)
        with torch_api.random.fork_rng():
            torch_api.manual_seed(seed)
            # Final bias cancels from pairwise margins; omit that unidentifiable parameter.
            self.net = nn.Sequential(
                nn.Linear(features, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False)
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def train_ranker(
    model: NeuralRanker,
    x: Tensor,
    relevance: Tensor,
    groups: Tensor,
    *,
    method: str = "pairwise",
    k: int = 3,
    steps: int = 80,
    lr: float = 0.03,
) -> list[float]:
    if steps < 1 or lr <= 0 or x.ndim != 2 or len(x) != len(relevance) or not len(x):
        raise ValueError("nonempty aligned features and positive training controls required")
    if not torch.isfinite(x).all():
        raise ValueError("features must be finite")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    model.train()
    for step in range(steps + 1):
        loss = pairwise_loss(model(x), relevance, groups, method=method, k=k)
        history.append(float(loss.detach()))
        if step == steps:
            break
        optimizer.zero_grad()
        cast(_Backward, loss).backward()
        cast(_Steppable, optimizer).step()
    return history
