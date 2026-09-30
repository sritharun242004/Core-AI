# Solution notes — controller behavior is a contract

## Worked budget trace

The notebook uses 3 model steps and 3 tool calls: two independent reads, one add,
and one final answer. The final answer uses a step but no tool call. A 2-call
budget rejects the third call without executing it. Retrying does not mint a new
budget. Plan-and-Execute charges one additional planning step: two subtasks with
one tool turn and one final each require 1 + 2×2 = 5 steps and 2 calls.

## Warmup answers

For tool latencies 40 and 70 ms, serial time is 110 ms; the ideal read-only fan-out
is 70 ms plus overhead. Real concurrency is not proof of determinism: callbacks
must read stable independent data. Returned observations are joined by request
order even if completion order changes. If B consumes A's output, parallelizing
A and B changes the computation; submit them on separate turns.

A budget of 2 calls with an attempted 3-call batch admits none. Schema-invalid
calls in an admitted batch count because attempted actions consume resources too.
A host-imposed call cap is enforceable without asking the model to be obedient.

## Build reference

`tests/test_agents.py` starts from capability and schema failures: unknown tool,
private lookup, bool-as-number, NaN, extra/missing keys, bounded magnitudes, wrong
output types and redacted callback exceptions. Controller tests cover empty or
ambiguous actions, exhaustion, zero budgets and global retry/planning limits.
A barrier forces two read callbacks to overlap; it will fail if fan-out becomes
serial. The ordered join test avoids timing-dependent expectations.

`Reflexion` stores trusted failure feedback and retries a scripted adapter. This
exercises feedback memory, not LLM-generated reflections or an improvement claim.
Its external verifier could also be wrong; version and test the verifier separately.
A failed answer followed by exhausted steps returns `step_budget`, not success.

## Interview solution

Start by defining the authority boundary: model text can request, but only the host
can authorize. Draw a state machine with terminal states and explicit budgets.
Separate schema correctness, tool safety and task correctness. Use structured
error codes to enable bounded recovery and make denied attempts visible in the
trajectory. Return a stop status when any global resource cap is exhausted.

For a long-horizon task, preserve versioned checkpoints, unresolved obligations,
source IDs and compact summaries. Never silently summarize away an approval or
change of principal. Treat retrieved instructions as untrusted data. A second
agent is another unreliable component, not an authorization authority.

## Deliberate limits / challenge direction

This reference has no persistent checkpoint store, natural-language model,
network transport, human approvals, tool timeout isolation, vector memory or
multi-agent framework. Do not attach an arbitrary shell to 'finish' the challenge.
Implement a deterministic checkpoint/replay extension using only public mock
records, or compare exact-match versus independent verifier evaluation on held-out
scripts. Label the extension honestly and retain the same global budget.
