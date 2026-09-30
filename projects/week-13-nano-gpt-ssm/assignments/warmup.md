# Warmup — derive one causal step (30–45 min)

1. For `d_model=8`, `heads=2`, and sequence length 4, draw the shape of Q, K,
   V, score, mask, probability, and head tensors.
2. Compute the scaled score matrix for two tokens by hand. Apply a strict
   upper-triangular future mask and show that every valid probability row sums
   to one.
3. Write the sinusoidal position values for positions 0 and 1 at dimensions
   0–3. Explain why position 0 begins with alternating 0 and 1.
4. Start with a failing test that changes only token 3 onward, then make the
   prefix logits invariant to that change.

**Deliverable:** a one-page shape table plus the two-token numerical attention
calculation. Include the axis over which softmax is taken and why normalizing
across heads or batches would be wrong.
