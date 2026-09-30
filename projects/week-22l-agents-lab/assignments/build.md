# Build — 2–3 hours

Run the executable notebook. Add an allowlisted in-memory `word_count` tool with
one bounded string argument and integer output. Register it from trusted host
code, not from a model-generated schema. Start with tests for empty text, extra
keys, oversized output and a non-string input.

Compare the same task under all three controllers. Script one incorrect answer,
then provide independent verifier feedback for Reflexion. Plan-and-Execute must
pay for its plan and share limits with every subtask. Record terminal status,
steps, calls, denied attempts and answer correctness as separate columns.

Deliver: tests, a notebook table, and a 300-word failure analysis. Acceptance:
no extra capabilities, no reset budgets during retry, no result order dependence
in independent parallel reads, and no claim that scripted success is an LLM score.
