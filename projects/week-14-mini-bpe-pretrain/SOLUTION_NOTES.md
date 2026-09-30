# Solution notes

- BPE merge order is part of the tokenizer contract. A tie-break that depends on dictionary insertion order can produce different IDs on another process.
- The tokenizer begins from Unicode characters to keep the derivation visible. It never crosses a document boundary while learning pair counts.
- RoPE rotates adjacent feature pairs. Its cache has one phase per position and pair; it is not an embedding table that must be learned.
- ALiBi adds a head-specific distance penalty to attention scores. Future tokens still need the causal mask.
- The language model shifts one token: input `x[t]` predicts target `x[t+1]`. A model that predicts the current token can achieve a misleadingly low loss.
- The checkpoint stores model configuration, weights, optimizer state, step, history, and optional tokenizer metadata. Always validate vocabulary compatibility on restore.

## What this does not prove

A tiny fixture can demonstrate tensor shapes and optimization behavior but cannot support scaling-law claims, data-quality claims, or benchmark comparisons. Replace it with a licensed corpus in the build assignment and log the tokenizer version with every checkpoint.
