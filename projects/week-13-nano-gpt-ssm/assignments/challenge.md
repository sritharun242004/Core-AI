# Challenge — make the comparison honest (3+ hrs)

Choose one:

- Implement a learned positional embedding and compare it with the fixed
  sinusoidal code under the same parameter/token budget.
- Add a valid multi-map/text split and a variance report across seeds; do not
  tune on the held-out result.
- Implement a chunked SSM scan and compare its outputs with the full scan before
  attempting optimization.
- Add a hybrid block that alternates a Transformer block and the explicit SSM,
  documenting precisely what is and is not borrowed from public hybrid-model
  descriptions.

**Required evidence:** tests first, a reproducible manifest, parameter count,
sequence length, optimizer, hardware/device, token budget, and failure cases.
Do not label the result Mamba, Jamba, or Griffin unless the implementation and
comparison are scoped to the specific public paper/release and the differences
are listed. A toy hybrid is a teaching artifact, not evidence about a private
production stack.
