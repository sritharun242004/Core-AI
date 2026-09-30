# Challenge — exact speculation and a serving experiment design (3+ hours)

1. Derive accepted probability mass `min(p,q)` and rejected mass. Show algebraically
   that adding normalized positive residual `(p-q)+` reconstructs p. Use
   p=(0.6,0.3,0.1), q=(0.2,0.3,0.5) and calculate the biased distribution obtained
   if rejection instead resamples from p. Handle identical distributions and q=0.
2. Extend the one-step sampler to two-token blocks on a tiny explicit transition
   table (not a downloaded model). Verify target conditionals for each draft prefix,
   stop at first rejection, discard the suffix, and emit the bonus target token when
   all proposals are accepted. Keep a token budget and deterministic seeded RNG.
3. Compare empirical length-2 sequence frequencies against exact target enumeration;
   use many seeds and justify a tolerance rather than asserting a lucky sample. Add
   tests for all-accepted, first-rejected, zero-support, and identical p/q cases.
4. Design, but do not automatically launch, a concurrency/arrival-rate sweep with
   matched prompts and output lengths. Define TTFT and TPOT SLOs, failure handling,
   goodput, memory and budget stop conditions. Include no-cache/warm-cache ablations.
5. Compare GPU serving with Groq, Cerebras, and SambaNova in a source-linked matrix:
   hardware family, offered model/version, request limits, region/network boundary,
   precision disclosure, and price unit. Mark unknowns. No unsourced “fastest” row.

**Deliver:** derivation, new sampler/tests, reproducible frequency table, and a
one-page experiment design. Mark all probability simulations and all vendor claims.

**Acceptance:** target distribution preserved, correct prefix-conditioned state,
no rejected suffix retained, no stochastic/greedy conflation, and a benchmark design
that separates network/provider/model changes from engine changes. GPU use is optional.
