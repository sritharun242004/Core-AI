# Build — reproduce the evidence ledger (2–3 hours)

Use the existing repository environment and the README commands. Work offline; do not download data or models.

## Deliverables

1. Run the percent notebook and save a small report of seed, library version, all DPO summary fields, all six safety metrics, held-out probe accuracy, and SAE train/test reconstruction plus activation statistics. Do not paste only a loss curve.
2. Add a failing regression test for a DPO bug of your choice: reversed preference sign, omitted reference, shared reference storage, or changed reference parameters. Restore the implementation and show the test passes. Verify a full reference state snapshot before/after training, not only `requires_grad=False`.
3. Evaluate an always-refuse callback and a callback that attempts an unlisted **mock** tool before refusing. Explain why neither can be reported as “100% safe” from required refusals alone. Retain case IDs and denied-call events.
4. Freeze ten new harmless paraphrases as a development extension. Include five allowed controls and five prohibited-by-toy-policy requests. Keep this separate from any final test suite; describe one failure of the narrow parser without calling it an LLM failure.
5. Fit the linear probe using training data only. Show train-derived normalization and evaluate on the independent heldout split once. Compare normal labels with the deliberately flipped-label negative control.
6. Train the SAE with the documented objective. Show that reconstruction improves, compare heldout MSE against the train-mean baseline, and report L0, L1, dead fraction and firing rates for the heldout codes. Explain why dictionary normalization is necessary.

**Acceptance:** actual parameter updates, no reference mutation, explicit metric denominators, deny-by-default mock tools, no test-label fitting, both reconstruction and activity metrics, and separate optimization/behavior/representation conclusions. Baseline fixtures must remain exactly 20 prompts; do not silently replace them with your development set.

**Boundary:** the red-team parser is separate from the categorical DPO policy. Do not portray parser results as a measured DPO safety improvement. See `SOLUTION_NOTES.md` for the worked rubric and common traps.
