# Build — 2–3 hours

Run the notebook and tests offline. Add a read-only public fixture tool to the MCP
teaching server: update its advertised schema and test invalid params, execution
errors, malformed fake responses and call-budget exhaustion before implementation.
Do not add shell/network/file access or imply full MCP conformance.

Add a third in-memory repository task with a protected reference file and an
independent verifier. Include a missing attempt, incorrect patch, forbidden path,
budget excess and correct answer with a denied attempt. Report denominators,
status reasons and token/call totals rather than a single success number.

Deliver: tests, executable trace and one-page evaluation card. Acceptance: fake
transports only, no code execution from patches, no dropped failures, and explicit
'local fixture suite, not SWE-Bench' labeling. Compare with SOLUTION_NOTES afterward.
