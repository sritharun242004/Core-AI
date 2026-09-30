# Worked solutions and failure modes

Read after attempting the assignments. These notes reveal the reference reasoning,
not hidden company practices or measured accelerator performance.

## Warmup — denominators and quantization

For `s=0`, tokens at `0.2, 0.3, 0.4`, completion at `0.5` seconds: TTFT=0.2 s,
TPOT=(0.4−0.2)/(3−1)=0.1 s, latency=0.5 s. A second request submitted at 0.1 with
one token at 0.6 has TTFT=0.5 s and **undefined** TPOT. Its completion at 0.8 does
not create a decode interval. Across an explicitly declared [0,1] window, 4 output
tokens produce 4 tokens/s. Mean TTFT is (0.2+0.5)/2=0.35 s. A no-token completion
has no TTFT and still counts in completed request throughput.

Do not divide tokens by summed request durations for overlapping arrivals: that
changes the denominator and destroys the server-throughput interpretation. Do not
call inverse weighted TPOT the server throughput: the first token, prefill, queue,
idle periods, completion, and parallel requests make them different quantities.

Group `[-1,0,0.5,1]` with INT4 uses qmax=7 and scale=1/7. In exact arithmetic,
ties-to-even gives codes `[-7,0,4,7]`: reconstruction of 0.5 is 4/7, giving error
1/14. Float32 representation of 1/7 can put that midpoint just below the tie and
produce code 3, reconstructing 3/7 instead; its error is also approximately 1/14.
The notebook reports the actual float32 codes rather than promising exact tie behavior. There are 16 possible 4-bit patterns, but this symmetric
scheme uses **15 values −7…7**, deliberately not −8. INT8 similarly uses −127…127.
Smaller groups localize outliers but add scales. For 128 weights, INT4 with groups of
64 needs 64 ideal packed bytes + 2×4 scale bytes = 72 bytes, versus 256 FP16 bytes:
3.56×, not exactly 4×. The NumPy codes are int8, so the implementation itself is not
a 72-byte packed array. Zero groups get scale=1 to avoid 0/0.

## Build — ownership, cache identity, fair reports

Three pages of four tokens hold A=5 and B=4 with tail waste 3. Appending 4 to A would
need one more page and must fail atomically: do capacity checking before editing
page tables or free sets. Appending 3 fits A's existing tail. Free B, then append 1
to A; page 2 can be reused and A now owns all three pages. At every step:

1. allocated page IDs are unique and disjoint from free IDs;
2. their union equals all physical page IDs;
3. request page count equals ceil(tokens/page size);
4. releasing each live request restores full capacity;
5. zero-token allocation has no pages, and double-free is an error.

This does not implement PagedAttention: that also needs a kernel to gather blocks,
shared-block reference counts, immutable cached prefixes, copy-on-write for partial
blocks, scheduler policies, and safe eviction. Sharing a page without refcounting
would turn `release()` into a use-after-free bug. Test those before adding sharing.

An exact key includes the token sequence and its computation namespace. `[1,23]`
and `[12,3]` must never collide via concatenation. Tenant, weights, tokenizer,
adapter, positional/mask setup, and dtype changes all invalidate reuse. Python's
process-randomized `hash()` is not a persisted cache key. Hashing decoded text can
lose token distinctions. Untrusted caller-supplied tenant IDs are not authorization.
Changing temperature alone does not change a deterministic prefill KV cache, but
changing tokens or a model adapter does. Stochastic inference would require a more
careful state contract. Multimodal inputs need their own content/revision identity.

For benchmarks, record actual output length rather than `max_new_tokens`. Warmup
must be excluded from both numerator and denominator. Fail visibly on an engine
error. Lower p95 from 3 samples is not robust evidence. A serial harness is useful
for debugging per-request latency and cannot establish continuous-batching gains.
When adding load, report rejected/timed-out requests, achieved arrival rate, queue
latency, goodput, and the exact wall window; do not silently filter failures.

## Challenge — why the residual is mandatory

Let target p=(0.6,0.3,0.1), draft q=(0.2,0.3,0.5). Propose x~q and accept with
min(1,p(x)/q(x))=(1,1,0.2). Accepted mass is min(p,q)=(0.2,0.3,0.1), total 0.6.
Rejection mass is 0.4. The normalized positive residual is (1,0,0), so final mass
is (0.2,0.3,0.1)+0.4×(1,0,0)=p. Resampling directly from p on rejection instead
would give (0.44,0.42,0.14), **not p**. Absolute differences are also wrong.

When q(x)=0, that x is never proposed; define its acceptance probability as 1 for
a finite vector API, and let positive residual mass recover any p(x)>0 there.
When p=q, rejection probability is zero; returning p as the unused residual is a
well-defined fallback. Normalization can introduce roundoff near 1e-16, so the
analytic test uses a tight absolute tolerance, not exact equality to floating zero.
The seeded Monte Carlo test checks target frequencies and acceptance≈0.4 for its
separate p=(0.7,0.2,0.1), q=(0.1,0.2,0.7) fixture, with a bounded tolerance.

For a multi-token block, draft sequentially, verify target conditionals for each
proposed prefix in a parallel target pass, accept until first rejection, draw the
residual at that position, and discard the unverified draft suffix. If all gamma
proposals are accepted, sample one extra target token. Update both caches to the
accepted prefix. Every p and q must be the **post-temperature/post-filtering**
distribution on the same vocabulary. Greedy token matching is a different procedure
and must not be presented as the stochastic acceptance proof.

A rough speed model is `(accepted tokens + correction/bonus) / (draft time + verify
+ scheduling/cache overhead)`, not acceptance rate alone. Large batches, a slow
draft, mismatched tokenizers, or frequent rejection can erase the benefit. This
project proves one-step distribution preservation only; it reports no speedup.

## Interview worked answer: 2× throughput, worse chat latency

First ask for the workload, output lengths, cache state, offered load, hardware,
precision, and timing boundary. Inspect TTFT, TPOT, end-to-end percentiles, queue
time, failures, and goodput against explicit SLOs. Bigger continuous batches can
amortize weight reads and raise throughput while requests wait longer or each
iteration gets slower. Chunked prefill can reduce decode stalls but has scheduling
tradeoffs. Quantization may buy memory and larger batches; a quality regression or
unsupported kernel may defeat the gain. Prefix caching primarily removes repeated
prefill work, not the cost of novel output tokens. Speculation trades extra compute
and memory for fewer serial target steps.

Baseline signal: correct denominators and no invented benchmarks. Senior signal:
controlled ablations and privacy-aware cache invalidation. Staff signal: admission
control, per-tenant isolation, overload behavior, failure-inclusive SLO goodput,
reproducibility and spend/teardown ownership. Vendor numbers remain vendor claims
unless independently reproduced on matching model and workload.
