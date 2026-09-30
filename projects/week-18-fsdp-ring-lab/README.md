# Week 18 — FSDP and ring-attention lab

Prove numerical invariants **before** renting accelerators. The offline reference uses
synthetic CPU tensors to account for sharded state, reproduce data-parallel gradients,
and compute exact causal attention in rotating K/V-block order. An optional CUDA
`torchrun` entrypoint exercises actual FSDP or DeepSpeed on a tiny model.

**Compute tier: red for the distributed extension; the complete correctness lab is
free and CPU-only.** No models, datasets, credentials, or network calls are needed.
Nothing here is a measured speedup claim. Read [COMPUTE.md](COMPUTE.md) before provisioning.

## Run locally

From the repository root, using the existing environment without changing the workspace lock:

```bash
ROOT="$PWD"
cd projects/week-18-fsdp-ring-lab
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1
"$ROOT/.venv/bin/python" -m pytest tests
"$ROOT/.venv/bin/python" notebooks/01-fsdp-ring.py
"$ROOT/.venv/bin/ruff" check src tests notebooks
"$ROOT/.venv/bin/ruff" format --check src tests notebooks
"$ROOT/.venv/bin/python" -m fsdp_ring_lab.integration --help
```

Standalone users need Python 3.13+, PyTorch 2.6+, and pytest. The notebook is percent
format (`# %%`), readable in an editor and executable as Python. Jupytext conversion
is optional, not a dependency of this lab. `data/README.md` documents fixture provenance.

## Public API

| API | Contract |
|---|---|
| `partition(length, world_size)` | Contiguous rank-ordered ceil-size shards; metadata includes valid row count; zero/empty ranks allowed. |
| `shard_rows`, `gather_rows` | Fixed-width first-dimension transport, trailing zero pad, shape/dtype/device validation, unpad on gather. |
| `memory_ledger(P, W, stage=...)` | Per-rank persistent bytes by component, optional full materialization buffer and activation allowance. |
| `collective_bytes(S, W, kind)` | Ideal ring sent/received bytes, **S is the full padded buffer**, not the local shard. |
| `zero_communication(P, W, stage=...)` | Explicit one-step collective schedule with configurable weight/gradient byte widths. |
| `ring_forward_bytes(...)` | Analytic uniform K/V forward rotations only; no backward or protocol bytes. |
| `full_batch_gradients`, `data_parallel_gradients` | Global element-mean MSE, deterministic pointwise model; no `.grad` or weight mutation. |
| `dense_attention`, `ring_attention` | CPU float32/float64 `[B,H,N,D]`; K shape equals Q; V may have a different channel count. |

Attention supports causal or noncausal self-attention and per-example prefix `lengths`.
All-padded examples return exact zero with zero gradients. Global N must be positive;
a batch length of zero is legal. Floating masks, NaNs/infinities, score overflow,
incompatible dtypes, and invalid partitions are rejected. Arbitrary packed-sequence
segment masks, dropout, cross-attention offsets, and mixed-precision GPU kernels are
not implemented. Noncontiguous tensors work through PyTorch operations.

DP sums each local squared-error sum divided by the **global target-element count**.
Empty ranks contribute zero, never NaN. Unequal rank means must not be averaged equally.
Training BatchNorm/dropout are rejected; custom stateful, stochastic, or batch-coupled
modules are outside the contract. The implementation simulates synchronized replicas
sequentially; it does not pretend to launch CPU collectives.

## What the accounting assumes

Default mixed-precision Adam storage: weights 2, gradients 2, fp32 master weights 4,
fp32 moments 8 bytes/parameter. Stage 0 replicates all fields; stage 1 shards optimizer
state/master weights; stage 2 also shards gradients; stage 3 also shards parameters.
Sharded fields reserve `ceil(P/W)` elements. Full replicated fields reserve P. Each
real FSDP wrapping unit/bucket can add its own padding; the model is a flat buffer.

`accounted_bytes` is **not actual peak memory**. A stage-3 `largest_unit` adds a full
weight materialization buffer that may coexist with shards. Activations can be supplied,
but gradients during backward, prefetch, communication workspaces, fragmentation, and
framework bookkeeping remain excluded. Stage 3 does not guarantee a model will fit.
FSDP mixed precision need not allocate state like this Adam example; the optional FP32
smoke uses different widths (4-byte weights/grads/moments, no separate master copy).

Ideal ring communication counts **sent bytes per rank**; receive is separately reported
and cluster-sent counts each transfer once. Stage 0 all-reduces gradients. Stages 1/2
reduce-scatter gradients and all-gather updated parameters; stage 1 still reserves a
replicated gradient buffer. Stage 3 gathers parameters for forward, regathers for
backward when resharding, and reduce-scatters gradients. These are transparent schedules,
not a promise that every DeepSpeed/FSDP release uses the identical transport plan.
Gradient accumulation, TP/PP traffic, checkpointing, and optimizer offload are excluded.

## Real accelerator smoke (optional, not executed in local validation)

On a pre-approved **two-GPU CUDA/NCCL machine**, use the installed CUDA-compatible
PyTorch environment. From this project, with `PYTHONPATH="$PWD/src"`:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m fsdp_ring_lab.integration --engine fsdp --steps 3

# DeepSpeed must already be installed in a compatible CUDA image; no auto-install.
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m fsdp_ring_lab.integration --engine deepspeed --steps 3 \
  --zero-config configs/zero-stage2.json
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m fsdp_ring_lab.integration --engine deepspeed --steps 3 \
  --zero-config configs/zero-stage3.json
```

The entrypoint uses real **FSDP1 `FULL_SHARD`**, not FSDP2's `fully_shard` API. Each rank
uses distinct seeded data, identical starting weights, equal local batch sizes, and
FP32 Adam; loss and gathered weights are checked against a full-batch reference each
step. The optimizer is constructed after FSDP wrapping. Teardown destroys the process
group in `finally`; torchrun owns the worker lifetimes. CPU/MPS, missing NCCL, invalid
rank maps, and non-torchrun launches fail explicitly instead of silently falling back.

DeepSpeed is a lazy optional dependency, deliberately absent from the offline package.
Python/PyTorch/CUDA/DeepSpeed compatibility and compiled extensions must be validated in
the chosen GPU image. Configs use microbatch 4, accumulation 1, FP32, no offload, and no
communication overlap so the initial contract is simple. Global batch is `4 * W`.
The script passes a normal PyTorch Adam optimizer to DeepSpeed; the JSON must not add a
second optimizer. Stage 3 disables small-parameter persistence so even the tiny model
exercises gathering. See [configs/README.md](configs/README.md) for tuning boundaries.

The full-batch oracle is replicated on every GPU, which **invalidates memory benchmarking
of this smoke script**. Replace it with a separate correctness phase before profiling.
Real multi-GPU execution and DeepSpeed installation were not performed for this project.
There is no real networked ring-attention backend: ring order is simulated on CPU;
backward uses ordinary PyTorch autograd, retains intermediates, and is tested against
dense Q/K/V gradients. A production ring implementation needs communication and a custom
recomputing backward, not just this loop moved onto a GPU.

## Architecture and study path

D2 source and an accessible hand-authored SVG fallback live at
`apps/book/public/diagrams/week-18-fsdp-ring.{d2,svg}`. The D2 CLI was unavailable;
the SVG is **not** presented as generated output. Reproduction instructions are in
[diagrams/README.md](diagrams/README.md).

1. Complete `assignments/warmup.md`: a memory and wire-byte ledger.
2. Complete `assignments/build.md`: uneven DP and forward/backward attention contracts.
3. Complete `assignments/challenge.md`: design an accelerator experiment before spending.
4. Read [SOLUTION_NOTES.md](SOLUTION_NOTES.md) after attempting the exercises.
