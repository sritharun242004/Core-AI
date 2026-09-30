# DeepSpeed configuration contracts

These JSON files are consumed by `python -m fsdp_ring_lab.integration --engine deepspeed`.
They are real DeepSpeed configuration inputs, not a record of an executed GPU run.
Use a compatible CUDA/NCCL image; DeepSpeed is not installed by the offline project.

- Microbatch = 4 per rank, accumulation = 1. Omit `train_batch_size`: DeepSpeed infers
  global batch from world size. A mismatch with `--batch-size` is rejected.
- FP16/BF16 are disabled for the FP32 numerical smoke. The memory ledger's default
  mixed-precision assumptions are separate and must not be applied to this run blindly.
- The Python entrypoint supplies PyTorch Adam; do not add an optimizer JSON block.
- Stage 2 shards gradients and optimizer states; stage 3 also shards parameters.
- Buckets are 1,000,000 **elements**, not bytes. They are deliberately modest, not
  autotuned. Stage 3 has zero prefetch/persistence thresholds for inspectable behavior.
- No CPU/NVMe offload: it introduces host bandwidth, pinned-memory and storage concerns
  that this smoke does not test. No overlap: first establish correctness, then profile.

For larger models, independently validate initialization memory (e.g. `deepspeed.zero.Init`),
checkpoint gather/consolidation memory, activation checkpointing, dynamic loss scaling
for fp16, dtype support, and optimizer compatibility. A JSON stage number alone does
not solve these. Do not copy tuning values from a frontier-model report into a tiny run.

Primary guide: https://www.deepspeed.ai/tutorials/zero/
