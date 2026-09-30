# Challenge — a defensible systems experiment (3+ hr)

Choose **one** branch. Cloud spending is never required to pass the offline branch.

## A. Offline backward design

Write a custom autograd attention function that recomputes blocks rather than retaining
every score/probability block. First add failing tests for gradients of Q, K, and V;
then derive the rowwise softmax backward and how multiple query owners sum dK/dV.
Save only the required inputs/output and per-row log-sum-exp. Compare dense gradients,
finite differences, padding, and future masking. Inventory saved tensors analytically;
do not equate their total with an allocator or CUDA peak.

## B. Approved real two-GPU smoke and profiling plan

Read `COMPUTE.md`, obtain budget approval, and run the actual FSDP entrypoint, then
DeepSpeed ZeRO-2/3 in a compatible image if available. Keep effective batch fixed.
Capture versions, topology, config, per-step loss and weight-equivalence records,
actual invoice, and teardown proof. Failed compatibility checks are valid findings;
never replace them with an invented successful run.

Design a **separate** benchmark without the replicated full-batch oracle. Include
warmup, CUDA synchronization, rank-max timings, at least five timed repetitions,
mean/dispersion, allocated/reserved peak memory, fixed useful tokens and effective
batch, and measured versus estimated byte columns. Discuss why a tiny MLP cannot
establish long-context transformer scaling. No advertised speedup is an acceptance
criterion.

**Rubric (25 points each):** numerical/partition correctness, communication and memory
accounting, reproducibility/limitations, cost and teardown discipline. Reference notes
explain the math; neither branch has a hidden claim of an already-run cloud benchmark.
