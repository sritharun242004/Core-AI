"""Inspectable positional mechanisms for small causal Transformers."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    """Rotate adjacent feature pairs: ``(x0,x1) -> (-x1,x0)``."""

    if x.shape[-1] % 2:
        raise ValueError("the final dimension must be even for a rotary embedding")
    first = x[..., ::2]
    second = x[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(start_dim=-2)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    *,
    base: float = 10_000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Return duplicated cosine/sine tables with shape ``(seq_len, head_dim)``."""

    if seq_len <= 0 or head_dim <= 0 or head_dim % 2:
        raise ValueError("seq_len and head_dim must be positive and head_dim must be even")
    if base <= 1:
        raise ValueError("base must be greater than one")
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
        * (-math.log(base) / head_dim)
    )
    angles = positions * frequencies
    # Interleave each frequency for the adjacent pair representation used by
    # rotate_half. Keeping the cache in float32 avoids losing phase precision.
    cos = torch.cos(angles).repeat_interleave(2, dim=-1).to(dtype=dtype)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1).to(dtype=dtype)
    return cos, sin


def _broadcast_rope(table: Tensor, x: Tensor, offset: int) -> Tensor:
    if table.ndim == 2:
        table = table[offset : offset + x.shape[-2]]
        if table.shape != (x.shape[-2], x.shape[-1]):
            raise ValueError("RoPE cache does not cover the input sequence")
        shape = [1] * (x.ndim - 2) + [x.shape[-2], x.shape[-1]]
        table = table.reshape(shape)
    elif table.ndim == 4:
        table = table[..., offset : offset + x.shape[-2], :]
    else:
        raise ValueError("RoPE tables must have shape (time, head_dim) or (..., time, head_dim)")
    return table.to(device=x.device, dtype=x.dtype)


def apply_rope(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, *, offset: int = 0
) -> tuple[Tensor, Tensor]:
    """Apply one cached rotary phase to query and key tensors.

    Query and key may be ``(batch, heads, time, head_dim)`` or any tensor whose
    final two axes are time and head features. Both tensors must have matching
    shape and an even head dimension.
    """

    if q.shape != k.shape or q.ndim < 3:
        raise ValueError("q and k must have the same rank and shape")
    if q.shape[-1] % 2:
        raise ValueError("the final dimension must be even for a rotary embedding")
    cos_q = _broadcast_rope(cos, q, offset)
    sin_q = _broadcast_rope(sin, q, offset)
    cos_k = _broadcast_rope(cos, k, offset)
    sin_k = _broadcast_rope(sin, k, offset)
    return (q * cos_q + rotate_half(q) * sin_q, k * cos_k + rotate_half(k) * sin_k)


def rotary_embedding(
    q: Tensor,
    k: Tensor,
    *,
    base: float = 10_000.0,
    offset: int = 0,
) -> tuple[Tensor, Tensor]:
    """Build a cache for ``q``/``k`` and apply it in one call."""

    cos, sin = build_rope_cache(
        offset + q.shape[-2], q.shape[-1], base=base, device=q.device, dtype=q.dtype
    )
    return apply_rope(q, k, cos, sin, offset=offset)


def alibi_slopes(n_heads: int) -> Tensor:
    """Return deterministic head slopes used by the ALiBi bias.

    This follows the simple geometric schedule from the ALiBi intuition: heads
    receive different distance penalties, while the helper stays transparent
    enough to reproduce by hand. It is a teaching implementation, not a claim
    about every production ALiBi variant.
    """

    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    return torch.pow(2.0, -8.0 * torch.arange(1, n_heads + 1, dtype=torch.float32) / n_heads)


def build_alibi_bias(
    n_heads: int,
    seq_len: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build additive ALiBi distances with shape ``(1, heads, time, time)``.

    Future locations are zero here and should be excluded by the causal mask;
    assigning them ``-inf`` in this helper would make the position bias less
    reusable for inspection and non-causal experiments.
    """

    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    slopes = alibi_slopes(n_heads).to(device=device, dtype=dtype).view(1, n_heads, 1, 1)
    positions = torch.arange(seq_len, device=device)
    distances = positions.view(1, 1, seq_len, 1) - positions.view(1, 1, 1, seq_len)
    distances = distances.clamp_min(0).to(dtype)
    return -slopes * distances


make_alibi_bias = build_alibi_bias
build_rope = build_rope_cache
