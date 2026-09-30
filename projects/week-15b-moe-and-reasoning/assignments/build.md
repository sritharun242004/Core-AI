# Build — sparse routing and a verifier loop

1. Run the offline notebook and annotate every shape from input to expert
   output. Inspect `Routing.dispatch_mask` with capacity factors `0.5`, `1.0`,
   and `2.0`.
2. Train `TinyMoEClassifier` on the fixed synthetic separator. Plot or print
   the loss, auxiliary loss, expert loads, and dropped-assignment fraction.
   Change only the capacity factor and explain the resulting trade-off.
3. Implement one deterministic arithmetic prompt batch. Have several sampled
   responses include both correct and incorrect final numbers. Compute binary
   rewards, group-relative advantages, and one `grpo_step`.
4. Compare one sample with majority vote over 3, 5, and 9 samples. Record
   pass/fail, number of verifier calls, seed, and tie behavior.
5. Write an experiment note containing seed, model dimensions, optimizer,
   capacity, reward definition, group size, and device. State why its result
   is a plumbing check rather than evidence of general reasoning.

**Acceptance checks:** `pytest`, Ruff, and the notebook pass offline; no
network download appears in code or output; the supervised loss reduces; all
routing and advantage assertions remain true.
