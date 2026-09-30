# Warm-up — one ledger, no accelerator (30–45 min)

1. With P=100, W=4, bf16 weights/gradients, fp32 master weights and two fp32 Adam
   moments, derive all four persistent per-rank storage totals. Run `memory_ledger`
   only after your hand calculation. State every excluded temporary component.
2. Repeat stage 3 for P=7,W=3. Draw valid/padded slots on each rank. Explain why a
   fractional parameter is not a valid allocation.
3. Derive sent bytes/rank for an all-reduce of a full padded 120-byte buffer on four
   ranks. Distinguish sent, received, and cluster-sent volume. Give the W=1 result.
4. Write a failing test that rejects an invalid world size or an inconsistent gather
   shape. Explain why silently trimming an arbitrary shard layout is dangerous.

**Submit:** a four-row memory table, one wire-volume diagram, a padding example, and
the failing-then-passing test transcript. Do not report predicted bytes as measured
GPU usage. Reference reasoning: `SOLUTION_NOTES.md` sections 1–2 (spoilers).
