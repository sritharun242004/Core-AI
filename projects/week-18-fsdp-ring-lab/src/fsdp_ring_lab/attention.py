"""Differentiable CPU ring-order simulation; not a fused distributed kernel.

Q/K/V are [batch, heads, sequence, channels]. Float32/float64 only. Prefix
lengths mask both keys and queries; empty rows have output and gradient zero.
The forward never builds a global score matrix in ring_attention, but PyTorch
retains block intermediates for backward. No training-memory saving is claimed.
"""

import math

import torch
from torch import Tensor

from .partition import integer, partition


def _validate(q: Tensor, k: Tensor, v: Tensor, lengths: Tensor | None, causal: bool) -> Tensor:
    if any(t.ndim != 4 or any(d < 1 for d in t.shape) for t in (q, k, v)):
        raise ValueError("Q/K/V must have nonempty [B,H,N,D] dimensions")
    if q.shape != k.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("Q/K must match; V must share batch, heads, sequence")
    if q.dtype not in (torch.float32, torch.float64) or any(t.dtype != q.dtype for t in (k, v)):
        raise ValueError("use one shared float32 or float64 dtype")
    if any(t.device.type != "cpu" for t in (q, k, v)):
        raise ValueError("this reference implementation is CPU-only")
    if not isinstance(causal, bool) or not all(torch.isfinite(t).all() for t in (q, k, v)):
        raise ValueError("causal must be bool and Q/K/V must be finite")
    batch, _, length, _ = q.shape
    if lengths is None:
        return torch.full((batch,), length, dtype=torch.long)
    if (
        lengths.shape != (batch,)
        or lengths.device.type != "cpu"
        or lengths.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("lengths must be a CPU integer tensor of shape [batch]")
    if torch.any(lengths < 0) or torch.any(lengths > length):
        raise ValueError("lengths must be between zero and sequence length")
    return lengths


def _allowed(q_indices: Tensor, k_indices: Tensor, lengths: Tensor, causal: bool) -> Tensor:
    mask = (q_indices[None, :, None] < lengths[:, None, None]) & (
        k_indices[None, None, :] < lengths[:, None, None]
    )
    if causal:
        mask = mask & (k_indices[None, None, :] <= q_indices[None, :, None])
    return mask[:, None, :, :]


def _scores(q: Tensor, k: Tensor) -> Tensor:
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if not torch.isfinite(scores).all():
        raise ValueError("QK score overflow: rescale inputs or use higher precision")
    return scores


def dense_attention(
    q: Tensor, k: Tensor, v: Tensor, *, lengths: Tensor | None = None, causal: bool = True
) -> Tensor:
    """Independent dense torch.softmax oracle, including all-masked rows."""
    lengths = _validate(q, k, v, lengths, causal)
    indices = torch.arange(q.shape[-2])
    allowed = _allowed(indices, indices, lengths, causal)
    scores = _scores(q, k).masked_fill(~allowed, -torch.inf)
    # Softmax(all -inf) is NaN: give empty rows finite dummy logits, then zero
    # every disallowed probability. No padded token can contribute to a loss.
    nonempty = allowed.any(dim=-1, keepdim=True)
    safe_scores = torch.where(nonempty, scores, torch.zeros_like(scores))
    weights = torch.softmax(safe_scores, dim=-1).masked_fill(~allowed, 0)
    return weights @ v


def ring_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    world_size: int = 2,
    query_block_size: int = 32,
    lengths: Tensor | None = None,
    causal: bool = True,
) -> Tensor:
    """Keep Q local; consume each K/V shard once in (rank-hop) mod W order.

    Rank order and absolute sequence indices are separate: causal masking uses
    absolute indices, never a local triangular mask. Empty transport ranks are
    valid. Padding is implicit in metadata, rather than physically transmitted.
    """
    lengths = _validate(q, k, v, lengths, causal)
    integer(query_block_size, "query_block_size", minimum=1)
    specs = partition(q.shape[-2], world_size)
    outputs = []
    for query_rank in specs:
        query_end = query_rank.start + query_rank.valid_size
        for start in range(query_rank.start, query_end, query_block_size):
            end = min(start + query_block_size, query_end)
            local_q = q[:, :, start:end]
            q_indices = torch.arange(start, end)
            shape = (*local_q.shape[:-1], 1)
            maximum = q.new_full(shape, -torch.inf)
            denominator = q.new_zeros(shape)
            numerator = q.new_zeros((*local_q.shape[:-1], v.shape[-1]))
            for hop in range(world_size):
                owner = specs[(query_rank.rank - hop) % world_size]
                if owner.valid_size == 0:
                    continue
                stop = owner.start + owner.valid_size
                allowed = _allowed(q_indices, torch.arange(owner.start, stop), lengths, causal)
                scores = _scores(local_q, k[:, :, owner.start : stop]).masked_fill(
                    ~allowed, -torch.inf
                )
                # Detaching the stabilizing shift is exact: the shift cancels
                # in numerator/denominator, just as in stable dense softmax.
                next_maximum = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True)).detach()
                safe_maximum = torch.where(torch.isfinite(next_maximum), next_maximum, 0)
                old_scale = torch.exp(maximum - safe_maximum)
                probabilities = torch.exp(scores - safe_maximum)
                numerator = old_scale * numerator + probabilities @ v[:, :, owner.start : stop]
                denominator = old_scale * denominator + probabilities.sum(dim=-1, keepdim=True)
                maximum = next_maximum
            safe_denominator = torch.where(denominator > 0, denominator, 1)
            outputs.append(numerator / safe_denominator)
    return torch.cat(outputs, dim=-2)
