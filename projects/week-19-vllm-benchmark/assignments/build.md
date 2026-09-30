# Build — an honest inference report (2–3 hours)

## Required offline work

1. Exercise a 3-page allocator with 4-token pages: allocate A=5, B=4; fail an append
   of 4 to A; append 3 to A; release B; append 1 to A; release A. Show capacity,
   ownership, and internal waste after every operation. Keep failure atomic.
2. Write a regression test for a tenant, adapter, and tokenizer change. An exact
   prefix hit must survive identical inputs and miss on each change. Demonstrate
   `[1,23] != [12,3]`, and that an extended prompt is not an exact-entry hit.
3. Build two synthetic trace scenarios with equal tokens but different queue/prefill
   and decode timings. Report TTFT p50/p95, macro and weighted TPOT, output tokens/s,
   complete window, and sample counts. Explain the different conclusions.
4. Write a threat model: cache timing leakage, unsafe shared pages, retention,
   cross-tenant keys, poisoned caller-supplied namespaces, and logs/hashes.

## Optional hardware extension (only with budget approval)

Follow `COMPUTE.md` and run the local-path CLI for HF and vLLM on the **same** GPU,
checkpoint, prompts/tokenizer, output limits, and precision. Use fresh processes,
cache disabled, and at least 10 repeats after warmup; then separately measure a
labelled warm prefix-cache condition. This is serial, not a server-throughput claim.
Do not acquire a model or provision compute as a side effect of running the tests.

**Deliver:** tests, two clearly labelled fixture JSON reports, a 400-word comparison,
and privacy/compute checklist. If hardware ran, append measured JSON and teardown
receipt; otherwise write “hardware experiment not run” rather than blank results.

**Acceptance:** full capacity restored, no aliasing or partial mutation on OOM,
all security identity changes miss, denominators auditable, and no invented speedup.
