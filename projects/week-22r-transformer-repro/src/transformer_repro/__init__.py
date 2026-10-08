"""Mechanism-level Transformer reproduction; not the WMT paper experiment."""

import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import cast

import torch
from torch import Tensor, nn


def attention(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Single causal self-attention over arbitrary leading batch dimensions."""
    if (
        q.ndim < 2
        or q.shape != k.shape
        or v.ndim != q.ndim
        or v.shape[:-1] != q.shape[:-1]
        or min(q.shape[-2:]) < 1
    ):
        raise ValueError("Q/K must match and V must share non-channel dimensions")
    if any(
        not value.is_floating_point()
        or value.dtype != q.dtype
        or value.device != q.device
        or not torch.isfinite(value).all()
        for value in (q, k, v)
    ):
        raise ValueError("Q/K/V must be finite floating tensors with a shared dtype/device")
    accumulation = torch.float64 if q.dtype == torch.float64 else torch.float32
    scores = q.to(accumulation) @ k.to(accumulation).transpose(-1, -2) / math.sqrt(q.shape[-1])
    if not torch.isfinite(scores).all():
        raise ValueError("attention score overflow")
    mask = torch.ones(q.shape[-2], q.shape[-2], device=q.device, dtype=torch.bool).triu(1)
    weights = scores.masked_fill(mask, -torch.inf).softmax(-1)
    return (weights @ v.to(accumulation)).to(v.dtype), weights


class TinyTransformer(nn.Module):
    """One pre-norm decoder block, learned positions; an explicit paper deviation."""

    def __init__(
        self, vocab_size: int = 8, width: int = 16, heads: int = 2, context: int = 16
    ) -> None:
        super().__init__()
        if min(vocab_size, width, heads, context) < 1 or width % heads:
            raise ValueError("positive dimensions and width divisible by heads required")
        self.heads, self.width, self.context = heads, width, context
        self.embedding = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context, width)
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        self.final_norm = nn.LayerNorm(width)
        self.unembed = nn.Linear(width, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.context:
            raise ValueError("tokens must be [batch, length] within context")
        batch, length = tokens.shape
        hidden = self.embedding(tokens) + self.position(torch.arange(length, device=tokens.device))
        q, k, v = self.qkv(self.norm1(hidden)).chunk(3, dim=-1)

        def heads(value: Tensor) -> Tensor:
            return value.reshape(batch, length, self.heads, self.width // self.heads).transpose(
                1, 2
            )

        attended, _ = attention(heads(q), heads(k), heads(v))
        hidden = hidden + self.output(attended.transpose(1, 2).reshape(batch, length, self.width))
        hidden = hidden + self.ffn(self.norm2(hidden))
        return self.unembed(self.final_norm(hidden))


def causal_batch(tokens: Tensor, *, context: int = 4) -> tuple[Tensor, Tensor]:
    if tokens.ndim != 1 or context < 1 or tokens.numel() <= context:
        raise ValueError("a 1D token stream longer than context is required")
    windows = tokens.unfold(0, context + 1, 1)
    return windows[:, :-1], windows[:, 1:]


def train_fixture(*, seed: int = 0, steps: int = 30, width: int = 16) -> list[float]:
    """Overfit a periodic synthetic corpus; held-out quality is deliberately not claimed."""
    if steps < 1:
        raise ValueError("steps must be positive")
    # PyTorch does not fully annotate fork_rng; this is its CPU-only call signature.
    with cast(Callable[[list[int]], AbstractContextManager[None]], torch.random.fork_rng)([]):
        torch.random.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        model = TinyTransformer(width=width)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
        x, y = causal_batch(torch.arange(40) % 8)
        history: list[float] = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x).reshape(-1, 8), y.reshape(-1))
            # These zero-argument PyTorch calls have incompletely typed optional arguments.
            cast(Callable[[], None], loss.backward)()
            cast(Callable[[], None], optimizer.step)()
            history.append(float(loss.detach()))
        return history


__all__ = ["TinyTransformer", "attention", "causal_batch", "train_fixture"]
