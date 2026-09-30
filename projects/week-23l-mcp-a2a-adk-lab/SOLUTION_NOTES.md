# Solution notes — separate the contracts

## Protocol traces

Initialization is request ID 1, then an initialized notification with **no reply**.
A list or call before the notification is rejected. A local missing public record
is a tool execution error in `result.isError`; malformed arguments are a JSON-RPC
invalid-params error. Clients validate both envelopes and the domain result shape.
If the transport swaps response IDs, refuse to interpret the body as this call's
answer. Stdio must contain only framed protocol data, never debug prints.

A2A here is `local-a2a-teaching-subset/0.1`, not a conformant implementation. An
explicit `demo/advance` tick makes submitted/working/terminal transitions testable.
Same request ID/task content can be retried idempotently; conflicting content
cannot overwrite an existing task. Canceled/completed/failed tasks are immutable.
The finite in-memory table is a resource boundary, not a durable queue.

## Numeric answers

Input tokens: 1,000 = 400 fresh + 400 cached reads + 200 writes. Output: 100.
Prices per million: fresh $2; cached $.50; writes $2.50 all-in; output $8.
Total = (400×2 + 400×.5 + 200×2.5 + 100×8) / 1,000,000 = **$0.0023**.
A hypothetical uniform half-price batch factor gives **$0.00115**. Real provider
batch/cache discounts may not stack that way; date and validate the actual rate card.

For a repeated 1,000-token prefix, plain fresh reads cost .002 each. Caching costs
.0025 for the first write plus .0005 for each later read. Break-even:
.0025 + (n−1)×.0005 < n×.002 → n > 4/3, so **two uses** already save money
under these illustrative assumptions. This ignores expiration/storage charges,
minimum prefix lengths, cache misses and changes to generated output.

Routing selects the cheapest model meeting *all* constraints. With a .8 quality
floor, the notebook's .7-quality cheap model is ineligible even when it costs less.
No route is better than an undetected downgrade for a protected quality threshold.
Quality here is a fixture scalar; deployment needs held-out task/slice estimates,
uncertainty, shift monitoring and explicit fallback criteria.

## Evaluation denominators

Both fixture patches can be correct while only one trajectory is safe: task
success = 2/2; safe success = 1/2; denied-attempt rate = 1/2. A missing attempt is
not dropped; it is a failed case. Two easy visible development fixtures cannot
support generalization, confidence or a SWE-Bench claim. All counters are provided
by the caller for this small harness; never trust model-written cost/denial logs.

The timeout verifier parses JSON and checks the exact integer, rejecting `true`.
The documentation verifier reads a version from the untouched package manifest;
it does not trust a text claim that the task passed. Only allowlisted existing
paths can change. No code supplied in a patch is evaluated or imported.

## Cache pitfalls

A model and prompt alone are not a safe cache key. Context/tool revisions, decoding
parameters and authenticated principal scope affect meaning and permissions. Use
least-privilege scope, not a tenant value provided by the model. Expiry occurs at
`now >= expires_at`; the injected monotonic clock makes the boundary deterministic.
A defensive copy prevents a consumer from corrupting another lookup. Failed
responses are not inserted. Cache hits can still be wrong when keys omit meaning.

## Interview solution and extension boundary

Draw separate boxes for model policy, tool protocol, agent delegation, execution
framework and evaluation. MCP exchanges capabilities/tool calls; A2A describes
agent-facing task interactions; ADK is application tooling. Neither a protocol
nor a framework supplies correctness, authentication or approval automatically.
Attach budgets, deadlines, provenance, redaction, idempotency and independent
verification to the execution host. Trace the request across every trust boundary.

A real SWE-Bench extension needs official pinned task data, reproducible container
images, safe resource isolation, patch application and held-out regression tests.
An Inspect AI extension must use its actual public APIs and versioned scorers.
Neither is in this reference; the framework comparison is conceptual. Optional
ADK execution is also not validated against cloud credentials. State these limits
rather than using a realistic-looking interface as proof of interoperability.
