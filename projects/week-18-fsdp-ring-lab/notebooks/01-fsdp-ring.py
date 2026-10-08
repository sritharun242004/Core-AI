# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportCallIssue=false, reportArgumentType=false
# %% [markdown]
# # Week 18 — invariants before accelerators
# Run from this project with PYTHONPATH=src python notebooks/01-fsdp-ring.py.
# All cells use local synthetic CPU tensors. No downloads, sockets, or credentials.
# The byte tables are analytic; they are not allocator measurements or timings.

# %%
import torch
from fsdp_ring_lab import (
    data_parallel_gradients,
    dense_attention,
    full_batch_gradients,
    gather_rows,
    memory_ledger,
    partition,
    ring_attention,
    ring_forward_bytes,
    shard_rows,
    zero_communication,
)
from torch import nn

torch.set_num_threads(1)
torch.manual_seed(18)
generator = torch.Generator().manual_seed(18)
print("OFFLINE CPU REFERENCE — no cloud or network execution")

# %% [markdown]
# ## 1. Persistent state and communication
# For 100 parameters and four ranks, 16-byte mixed-precision Adam state costs
# 1600 / 700 / 550 / 400 bytes per rank at stages 0 / 1 / 2 / 3.
# Real implementations vary in gradient dtype, padding, and master copies.

# %%
for stage, expected in enumerate((1600, 700, 550, 400)):
    ledger = memory_ledger(100, 4, stage=stage, largest_unit=40)
    communication = zero_communication(100, 4, stage=stage)
    assert ledger.persistent_bytes == expected
    print(
        f"stage={stage} persistent={ledger.persistent_bytes} "
        f"materialization={ledger.materialization_bytes} "
        f"sent/rank={communication.sent_per_rank} bytes"
    )

# %% [markdown]
# ## 2. Uneven transport never creates extra examples
# Seven rows on three ranks reserve nine slots: [3, 3, 1] valid rows.

# %%
rows = torch.arange(14).reshape(7, 2)
shards = shard_rows(rows, 3)
assert [spec.valid_size for spec in partition(7, 3)] == [3, 3, 1]
torch.testing.assert_close(gather_rows(shards, 7), rows)
print("partition valid counts:", [spec.valid_size for spec in partition(7, 3)])

# %% [markdown]
# ## 3. Data-parallel gradient equivalence
# Local sum / global element count is equivalent to sample-weighted local means.
# Averaging the three rank means equally would over-weight the one-row shard.

# %%
model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2)).double()
x = torch.randn(7, 3, dtype=torch.float64, generator=generator)
y = torch.randn(7, 2, dtype=torch.float64, generator=generator)
full = full_batch_gradients(model, x, y)
parallel = data_parallel_gradients(model, x, y, world_size=3)
for expected, actual in zip(full.gradients, parallel.gradients, strict=True):
    torch.testing.assert_close(expected, actual, atol=1e-12, rtol=1e-11)
print("full / simulated DP mean MSE:", full.loss, parallel.loss)

# %% [markdown]
# ## 4. Ring-order attention: forward AND backward
# Each Q owner sees K/V shards in rank-hop order, with absolute-position masks.
# Prefix lengths [7, 4] ensure padded queries/keys have no output/gradient.

# %%
q, k, v = (
    torch.randn(2, 2, 7, dim, generator=generator, dtype=torch.float64).requires_grad_()
    for dim in (3, 3, 4)
)
lengths = torch.tensor([7, 4])
ring = ring_attention(q, k, v, world_size=3, query_block_size=2, lengths=lengths)
dense = dense_attention(q, k, v, lengths=lengths)
torch.testing.assert_close(ring, dense, atol=1e-12, rtol=1e-11)
upstream = torch.randn(ring.shape, generator=generator, dtype=torch.float64)
ring_grads = torch.autograd.grad((ring * upstream).sum(), (q, k, v))
dense_grads = torch.autograd.grad((dense * upstream).sum(), (q, k, v))
for actual, expected in zip(ring_grads, dense_grads, strict=True):
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    assert torch.count_nonzero(actual[1, :, 4:]) == 0
print("max forward error:", (ring - dense).abs().max().item())
print(
    "max Q/K/V gradient errors:",
    [
        (actual - expected).abs().max().item()
        for actual, expected in zip(ring_grads, dense_grads, strict=True)
    ],
)

# %% [markdown]
# ## 5. Ring wire volume is not wall time
# Uniform padded K/V blocks rotate W-1 times. This excludes backward traffic,
# latency, overlap, causal load imbalance, and protocol overhead.

# %%
traffic = ring_forward_bytes(7, 3, batch=2, heads=2, key_dim=3, value_dim=4, element_bytes=8)
assert traffic.sent_per_rank == 1344
print("analytic ring forward sent/rank:", traffic.sent_per_rank, "bytes")
print("PASS: accounting, partitions, DP gradients, attention outputs + Q/K/V gradients")
print("Not measured: GPU memory, distributed speed, network overlap, production scaling")
