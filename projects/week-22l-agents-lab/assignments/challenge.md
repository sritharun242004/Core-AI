# Challenge — 3+ hours

Implement versioned checkpoint/replay for a ten-subtask **public mock** workflow.
Persist only JSON-safe observations, summary memory, outstanding subtask IDs and
budget counters. Never serialize executable callbacks or untrusted pickle data.
Use a local temporary directory only in the host checkpoint layer, not as an
agent-controlled filesystem tool.

Write tests for replay after interruption, changed tool version, changed principal,
unknown schema version and attempted budget reset. Compare full history with
bounded summaries on a held-out deterministic task set. Explain what information
a summary must retain to preserve correctness and authorization.

Deliver: tests, a failure matrix, exact version/budget provenance, and a one-page
long-horizon design. Do not attach shell/network tools or report this as a deployed
multi-agent system. Extension is learner work, not included in the reference.
