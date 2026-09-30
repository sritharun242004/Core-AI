# Challenge — add one multimodal boundary (3+ hrs)

Choose one extension and write tests before implementation:

- replace the alias dense fixture with a small local, explicitly licensed sentence/image
  embedding model, then measure recall and latency against the baseline;
- build a typed entity graph from a new fixture and compare lexical expansion with graph
  expansion, including a hop and fan-out budget;
- implement a local cross-encoder-like reranker and ablate candidate count;
- make late interaction tokenization and max-sim inspectable, then compare pooled cosine
  with token-level scores;
- add a tiny gradient update to the denoising predictor and report before/after loss on
  a fixed batch, while preserving deterministic seeds;
- replace fixture voice adapters with a local, documented ASR/TTS implementation and
  test transcript, citation, timeout, and no-evidence behavior.

For the final note, include an error taxonomy: retrieval miss, graph over-expansion,
unsupported generation, malformed audio, and evaluation blind spots. Name what your
experiment can establish locally and what it cannot establish about a hosted product or
any company's private multimodal stack.
