import pytest
import torch
from fsdp_ring_lab import (
    collective_bytes,
    gather_rows,
    memory_ledger,
    partition,
    ring_forward_bytes,
    shard_rows,
    zero_communication,
)


@pytest.mark.parametrize("length,world", [(0, 3), (1, 4), (7, 3), (12, 3)])
def test_partition_round_trip_and_padding(length, world):
    x = torch.arange(length * 2, dtype=torch.float64).reshape(length, 2)
    specs = partition(length, world)
    shards = shard_rows(x, world)
    assert len(specs) == len(shards) == world
    assert sum(s.valid_size for s in specs) == length
    assert all(s.shard_size == (length + world - 1) // world for s in specs)
    for s, shard in zip(specs, shards, strict=True):
        assert shard.shape == (s.shard_size, 2)
        assert torch.count_nonzero(shard[s.valid_size :]) == 0
    torch.testing.assert_close(gather_rows(shards, length), x)


@pytest.mark.parametrize("length,world", [(-1, 2), (4, 0), (4, -2), (True, 2), (4, 1.5)])
def test_partition_rejects_invalid_dimensions(length, world):
    with pytest.raises(ValueError):
        partition(length, world)


def test_gather_rejects_incompatible_layout():
    with pytest.raises(ValueError):
        gather_rows([torch.ones(2, 3), torch.ones(1, 3)], 3)
    with pytest.raises(ValueError):
        gather_rows([], 0)
    with pytest.raises(ValueError):
        gather_rows([torch.ones(2, 3)], 3)


@pytest.mark.parametrize("stage,expected", [(0, 1600), (1, 700), (2, 550), (3, 400)])
def test_mixed_precision_adam_memory(stage, expected):
    ledger = memory_ledger(100, 4, stage=stage, largest_unit=40, activation_bytes=17)
    assert ledger.persistent_bytes == expected
    assert ledger.materialization_bytes == (80 if stage == 3 else 0)
    assert ledger.accounted_bytes == expected + ledger.materialization_bytes + 17
    assert ledger.padded_parameters == 100


def test_memory_padding_and_single_rank():
    ledger = memory_ledger(7, 3, stage=3)
    assert ledger.persistent_bytes == 3 * 16
    assert ledger.padded_parameters == 9
    assert memory_ledger(7, 1, stage=3).persistent_bytes == 7 * 16
    assert memory_ledger(0, 3, stage=3).accounted_bytes == 0


@pytest.mark.parametrize(
    "kwargs", [{"stage": 4}, {"largest_unit": 11}, {"parameter_bytes": 0}, {"activation_bytes": -1}]
)
def test_memory_rejects_bad_contracts(kwargs):
    with pytest.raises(ValueError):
        memory_ledger(10, 2, **kwargs)


@pytest.mark.parametrize(
    "kind,factor", [("all_reduce", 2), ("all_gather", 1), ("reduce_scatter", 1)]
)
def test_ring_collective_byte_convention(kind, factor):
    result = collective_bytes(120, 4, kind)
    assert result.sent_per_rank == 90 * factor
    assert result.received_per_rank == result.sent_per_rank
    assert result.cluster_sent == 4 * result.sent_per_rank
    assert collective_bytes(120, 1, kind).sent_per_rank == 0


def test_zero_schedule_and_padding():
    assert zero_communication(100, 4, stage=0).sent_per_rank == 300
    assert zero_communication(100, 4, stage=1).sent_per_rank == 300
    assert zero_communication(100, 4, stage=2).sent_per_rank == 300
    assert zero_communication(100, 4, stage=3).sent_per_rank == 450
    assert zero_communication(100, 4, stage=3, reshard_after_forward=False).sent_per_rank == 300
    assert zero_communication(7, 3, stage=3).sent_per_rank == 36
    assert (
        ring_forward_bytes(
            7, 3, batch=2, heads=2, key_dim=4, value_dim=5, element_bytes=4
        ).sent_per_rank
        == 864
    )


@pytest.mark.parametrize(
    "payload,world,kind",
    [(7, 3, "all_gather"), (-4, 2, "all_reduce"), (4, 0, "all_reduce"), (4, 2, "broadcast")],
)
def test_collective_rejects_ambiguous_payload(payload, world, kind):
    with pytest.raises(ValueError):
        collective_bytes(payload, world, kind)
