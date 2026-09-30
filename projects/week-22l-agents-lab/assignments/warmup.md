# Warmup — 30 minutes

1. Trace a ReAct run containing two reads, an add and a final answer. Count model
   steps and tool attempts separately; include the rejected request to a private key.
2. For independent 40 ms and 70 ms reads, compute serial/parallel ideal latency.
   Explain why a dependent read cannot use that speedup formula.
3. Write failing tests for a bool passed to `add`, an extra argument and a malformed
   action containing both calls and a final answer. Run offline pytest.

Deliver: one trace table and three tests. Acceptance: failures occur at the host
boundary before callback execution; denied attempts remain visible and charged.
See `SOLUTION_NOTES.md` only after completing the arithmetic.
