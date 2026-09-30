# Solution notes — read after your attempt

## 1. Storage is not peak memory

For P parameters on W ranks, use p weight bytes, g gradient bytes, and o optimizer
bytes including a master copy. Ignoring ceil padding, stage 0 costs `(p+g+o)P`, stage 1
`(p+g)P + oP/W`, stage 2 `pP + (g+o)P/W`, stage 3 `(p+g+o)P/W`.

At P=100, W=4, p=g=2 and o=12, persistent bytes are 1600, 700, 550, 400.
At P=7, W=3, stage 3 reserves 3 elements per rank, not 7/3: 48 bytes, with 9 total
transport slots. A full 40-parameter materialization adds 80 bytes to the selected
stage-3 ledger. It does not magically replace persistent state or bound actual peak.
FSDP wrapping units and buckets each pad independently in real systems. With FP32 Adam
and no separate master copy, weights 4 + gradients 4 + two moments of 4 each = 16 too,
but the **stage-by-stage split differs** from bf16 plus master weights.

Adam states are lazily allocated by typical optimizers; measure after warmup/updates,
not only after constructing a model. Forward activations, backward gradients of an
unsharded unit, communication buffers, CUDA context, allocator reserve, and prefetch
can dominate the gap between a ledger and an OOM.

## 2. Wire bytes need a convention

For padded full-buffer size S and W ranks, ideal ring reduce-scatter and all-gather
each send `(W-1)S/W` bytes/rank. All-reduce contains both phases. Receiving the same
volume is not another set of network transfers in a cluster-sent total. Count each
wire transfer once, then report bidirectional NIC load separately if needed.

At P=100, W=4, p=g=2, stage 0 sends 300 bytes/rank. The stated stage-1/2 schedule sends
150 bytes of gradient reduce-scatter + 150 of updated-weight all-gather. Stage 3 with
reshard-after-forward sends 150 + 150 of weights + 150 of gradients = 450 bytes/rank.
Retaining gathered weights drops the second gather but changes memory residency.
These are analytic schedules, not exact NCCL packet counts or all-version DeepSpeed
behavior. The latency model `T ≈ number_of_steps * alpha + bytes / bandwidth` explains
why the same byte count with many tiny collectives can be much slower.

## 3. Uneven data-parallel losses

The intended loss is the mean across all output elements. Rank r with n_r examples
and fixed output dimension d computes `sum(squared_error_r)/(N*d)`, then rank gradients
sum. Equivalently weight its mean by n_r/N. For local sample counts [3,3,1], equal
rank means overweight the last sample by 7/3 relative to its intended coefficient.

Real DDP/FSDP average gradients across ranks, so unequal batches require each local
mean to be scaled by `W*n_r/N` before backward (all ranks must agree on N). For masked
language-model losses use **valid target-token counts**, not padded batch counts.
An empty rank still participates in required distributed collectives, unlike this
sequential simulator where its mathematical zero contribution is skipped. BatchNorm
and stochastic/dropout stream differences can invalidate equivalence; the reference
rejects the common built-ins and documents the deterministic pointwise-model contract.

## 4. Online softmax, including empty blocks

For one query row keep running maximum m, denominator l, and unnormalized vector u.
For block logits s, set `m_new=max(m,max(s))`, `a=exp(m-m_new)`, `p=exp(s-m_new)`;
then `l_new=a*l+sum(p)` and `u_new=a*u+p@V`. Finally output u/l. The rescaling keeps
all terms in a shared exponential coordinate system; averaging independently
normalized block outputs loses their relative partition-function weights.

Example: logits [0, log(2)] and scalar values [2,8], one key per block. First state is
m=0, l=1, u=2. Second m=log(2), a=1/2, p=1, so l=1.5, u=9, output=6. Dense probabilities
are [1/3,2/3], also yielding 6. If the second key is in the future, it contributes
nothing and output remains 2. If both are padding, output is defined as zero.

A causal rank can receive an entirely future block first. Using `-inf - -inf` gives
NaN; clamp the **shift** for empty state to zero, keep masked logits at -inf, and
use denominator 1 only when l=0. This gives zero exponentials/numerator. The max shift
is detached from autograd because it cancels algebraically in the ratio; gradients
through exponentials and the recurrence remain exact. Finite-difference gradcheck
and independent dense-softmax backward check this decision, not just output shape.

Backward reference equations, with upstream G, probabilities A and scores S:
`dV=A^T G`, `dA=G V^T`, `dS=A*(dA-row_sum(dA*A))`, `dQ=dS K/sqrt(d)`,
`dK=dS^T Q/sqrt(d)`. Padding/future positions have A=0. In a real ring, contributions
to dK/dV from every Q owner must return to the K/V owner and be summed, not overwritten.
Production kernels recompute tiles using saved log-sum-exp/output instead of retaining
every block's autograd graph. This project deliberately does not claim that optimization.

## 5. Rank order is not token order

Rank r sees K/V owners `(r-hop) mod W`. Use absolute indices for the causal comparison
`key_position <= query_position`. A local triangular mask fails for cross-rank blocks.
The transport reserves ceil(N/W) token slots, and lengths are prefix-valid positions
per example. Extra padded Q rows have zero output and zero loss gradient; padded K/V
rows have zero influence. For N=7,W=3, owner 2 has one valid token, not three. Empty
owners occur when W>N. Noncausal mode changes only the future mask, not padding.

Our ring walks actual tokens and models padded wire volume separately. With batch B,
heads H, local width c, K dimension d_k, V dimension d_v, element width b, an unoptimized
forward ring sends `(W-1)*B*H*c*(d_k+d_v)*b` bytes/rank. Local data is consumed first,
then W-1 rotations. No final return is necessary for forward. This is not the backward
communication formula. Causal masking alone does not remove network transfers.

## 6. Interview worked solution

**Problem:** Eight GPUs still OOM on a long-context model after enabling full sharding.

1. Establish parameters, dtype and master/moment policy, global/local sequence/batch,
   FSDP unit sizes, optimizer allocation point, and whether the OOM is forward/backward
   or checkpoint save. Fix a reproducible input and capture per-rank memory timeline.
2. Separate persistent state from materialized layer weights and activations. If the
   whole network is one wrapping unit, sharded-at-rest does not mean small working set.
   Use smaller units and bound prefetch; inspect peak activations and attention scores.
3. Activation checkpointing trades recomputation for saved activations. Blockwise
   attention avoids a global score matrix; context parallelism distributes sequence
   activation storage but adds K/V and gradient communication. TP splits operator
   work; PP splits layers but introduces pipeline bubbles. These are not synonyms.
4. Prove gradient equivalence on a tiny deterministic model, then validate loss curves
   on fixed data. Preserve token-normalized effective batch under accumulation.
5. Profile actual interconnect/overlap before choosing a larger W. Report repeated
   throughput at a fixed useful token count and maximum per-rank memory; keep compute
   estimates and observed results distinct. Terminate the paid resources afterwards.

**What surprised me:** reducing stored bytes can increase transferred bytes; exact
attention can be reordered but not independently normalized; a numerically correct
Python autograd implementation is not automatically a memory-efficient training kernel.
