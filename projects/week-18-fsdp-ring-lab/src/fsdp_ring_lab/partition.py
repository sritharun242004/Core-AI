"""Contiguous equal-size transport shards; padding never becomes training data."""

from dataclasses import dataclass

import torch
from torch import Tensor


def integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Shard:
    rank: int
    start: int
    valid_size: int
    shard_size: int
    global_size: int


def partition(length: int, world_size: int) -> tuple[Shard, ...]:
    """Allow empty ranks, including an empty global tensor; preserve rank order."""
    integer(length, "length")
    integer(world_size, "world_size", minimum=1)
    width = (length + world_size - 1) // world_size
    return tuple(
        Shard(rank, rank * width, max(0, min(width, length - rank * width)), width, length)
        for rank in range(world_size)
    )


def shard_rows(tensor: Tensor, world_size: int) -> tuple[Tensor, ...]:
    if tensor.ndim < 1:
        raise ValueError("tensor must have a row dimension")
    shards = []
    for spec in partition(tensor.shape[0], world_size):
        valid = tensor[spec.start : spec.start + spec.valid_size]
        padding = tensor.new_zeros((spec.shard_size - spec.valid_size, *tensor.shape[1:]))
        shards.append(torch.cat((valid, padding), dim=0))
    return tuple(shards)


def gather_rows(shards: tuple[Tensor, ...] | list[Tensor], length: int) -> Tensor:
    """Validate fixed-width transport, concatenate by rank, discard trailing pads."""
    if not shards or shards[0].ndim < 1:
        raise ValueError("at least one non-scalar shard is required")
    specs = partition(length, len(shards))
    first = shards[0]
    shape = (specs[0].shard_size, *first.shape[1:])
    if any(s.shape != shape or s.dtype != first.dtype or s.device != first.device for s in shards):
        raise ValueError("shards must share expected shape, dtype, and device")
    return torch.cat(tuple(shards), dim=0)[:length]
