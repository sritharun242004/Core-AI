"""Analytic storage and ideal ring byte counts, not measured memory or speed."""

from dataclasses import dataclass

from .partition import integer, partition


@dataclass(frozen=True)
class MemoryLedger:
    parameter_bytes: int
    gradient_bytes: int
    master_bytes: int
    moment_bytes: int
    materialization_bytes: int
    activation_bytes: int
    padded_parameters: int

    @property
    def persistent_bytes(self) -> int:
        return self.parameter_bytes + self.gradient_bytes + self.master_bytes + self.moment_bytes

    @property
    def accounted_bytes(self) -> int:
        """Selected components only: NOT allocator peak or an OOM guarantee."""
        return self.persistent_bytes + self.materialization_bytes + self.activation_bytes


@dataclass(frozen=True)
class Communication:
    sent_per_rank: int
    received_per_rank: int
    cluster_sent: int


def _stage(stage: int) -> None:
    integer(stage, "stage")
    if stage > 3:
        raise ValueError("stage must be 0 (replicated), 1, 2, or 3")


def memory_ledger(
    parameters: int,
    world_size: int,
    *,
    stage: int = 0,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    master_bytes: int = 4,
    moment_bytes: int = 8,
    largest_unit: int = 0,
    activation_bytes: int = 0,
) -> MemoryLedger:
    """Flat-buffer ZeRO storage; defaults: bf16 weights/grads + fp32 Adam.

    Sharded fields reserve ceil(P/W) elements. Stage 3 materialization counts
    a full largest-unit weight buffer coexisting with persistent shards. It
    omits gradient temporaries, prefetch, NCCL, fragmentation, and workspace.
    Frameworks differ in master-copy and mixed-precision policies: supply the
    byte widths actually used, including zero if no master weights exist.
    """
    _stage(stage)
    spec = partition(parameters, world_size)[0]
    for name, value in (("parameter_bytes", parameter_bytes), ("gradient_bytes", gradient_bytes)):
        integer(value, name, minimum=1)
    for name, value in (
        ("master_bytes", master_bytes),
        ("moment_bytes", moment_bytes),
        ("largest_unit", largest_unit),
        ("activation_bytes", activation_bytes),
    ):
        integer(value, name)
    if largest_unit > parameters:
        raise ValueError("largest_unit cannot exceed parameters")
    return MemoryLedger(
        (spec.shard_size if stage == 3 else parameters) * parameter_bytes,
        (spec.shard_size if stage >= 2 else parameters) * gradient_bytes,
        (spec.shard_size if stage >= 1 else parameters) * master_bytes,
        (spec.shard_size if stage >= 1 else parameters) * moment_bytes,
        largest_unit * parameter_bytes if stage == 3 else 0,
        activation_bytes,
        spec.shard_size * world_size,
    )


def collective_bytes(payload_bytes: int, world_size: int, kind: str) -> Communication:
    """Full logical padded buffer size, NOT local-shard size; ideal ring only.

    Sent and received are reported separately. Cluster sent counts each wire
    transfer once. Padding must make the full payload divisible by world size.
    """
    integer(payload_bytes, "payload_bytes")
    integer(world_size, "world_size", minimum=1)
    if kind not in {"all_reduce", "all_gather", "reduce_scatter"}:
        raise ValueError("unsupported collective kind")
    if payload_bytes % world_size:
        raise ValueError("payload_bytes must be padded to a multiple of world_size")
    factor = 2 if kind == "all_reduce" else 1
    sent = factor * (world_size - 1) * (payload_bytes // world_size)
    return Communication(sent, sent, world_size * sent)


def zero_communication(
    parameters: int,
    world_size: int,
    *,
    stage: int = 0,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    reshard_after_forward: bool = True,
) -> Communication:
    """One step, one gradient synchronization, no accumulation or overlap.

    Stage 0: gradient all-reduce. Stages 1/2: gradient reduce-scatter plus
    updated-weight all-gather (stage 1 still reserves full gradient storage).
    Stage 3: forward weight all-gather, optional backward weight all-gather,
    gradient reduce-scatter. Framework schedules may differ from this model.
    """
    _stage(stage)
    integer(parameter_bytes, "parameter_bytes", minimum=1)
    integer(gradient_bytes, "gradient_bytes", minimum=1)
    if not isinstance(reshard_after_forward, bool):
        raise ValueError("reshard_after_forward must be bool")
    padded = partition(parameters, world_size)[0].shard_size * world_size
    grad = collective_bytes(padded * gradient_bytes, world_size, "reduce_scatter").sent_per_rank
    weight = collective_bytes(padded * parameter_bytes, world_size, "all_gather").sent_per_rank
    if stage == 0:
        sent = 2 * grad
    elif stage in (1, 2):
        sent = grad + weight
    else:
        sent = grad + weight * (1 + int(reshard_after_forward))
    return Communication(sent, sent, world_size * sent)


def ring_forward_bytes(
    sequence_length: int,
    world_size: int,
    *,
    batch: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    element_bytes: int = 4,
) -> Communication:
    """W-1 uniform K/V rotations; local block first, no final return-to-owner.

    Counts an unoptimized causal ring too: masking is not network pruning.
    Backward K/V gradient transfers are deliberately NOT counted here.
    """
    for name, value in (
        ("batch", batch),
        ("heads", heads),
        ("key_dim", key_dim),
        ("value_dim", value_dim),
        ("element_bytes", element_bytes),
    ):
        integer(value, name, minimum=1)
    width = partition(sequence_length, world_size)[0].shard_size
    sent = (world_size - 1) * batch * heads * width * (key_dim + value_dim) * element_bytes
    return Communication(sent, sent, world_size * sent)
