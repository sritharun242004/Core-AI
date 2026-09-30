# Challenge — design a claim that could fail (3+ hours)

Choose one extension. Keep all inputs synthetic and local; no real secrets, destructive tools, provider calls or downloaded checkpoints are needed.

## A. Preference generalization and policy drift

Replace discrete context IDs with generated feature vectors whose preferred action follows a known rule. Partition by generation seed/source before creating preference pairs. Train on one split, choose β/steps on validation, and freeze the protocol before one final test evaluation. Report preference accuracy, chosen probability, pairwise margin, and exact categorical KL to the reference on identical contexts. Compare at least three seeds. Explain why smaller training loss need not mean better test performance or less drift.

## B. A trajectory-aware safety gate

Add a bounded multi-step mock workflow with two harmless tools and a human-approval simulation for a consequential toy action. Define threat boundaries and authorization separately from the assistant's prose. Test denied tool names, exact schemas, unauthorized resource keys, call-budget exhaustion, quoted-content controls, and a final refusal after an earlier denied attempt. Use Week 20-style paired case outcomes and preserve rollback criteria. Do not add a real shell/network tool or call Python callbacks a sandbox.

## C. Reconstruction–sparsity trade-offs

Generate independent train/validation/test activation splits. Sweep L1 coefficients `{0, .001, .01, .1}` on training, select a rule on validation, and evaluate that frozen choice once on test. Report elementwise MSE, train-mean baseline, mean L0/L1, feature firing rates and dead fraction over three seeds. Add a shuffled-label probe control on an independently generated probe dataset. Report feature permutation/duplication ambiguity; do not label learned coordinates by desired semantics without evidence.

## Required closing paragraph

State the strongest claim your experiment supports, one plausible alternative explanation, and the next measurement that would distinguish them. If proposing a causal interpretation, specify a controlled intervention, matched negative controls and downstream outcome before running anything. A proposal is not a causal result; this lab does not reproduce full Anthropic Circuits.

**Acceptance:** frozen split identities, a genuinely falsifiable comparison, explicit budgets and uncertainty, regression tests written before changed numerical code, no unsupported real-model safety claims. The implementation is a learner extension, not an included or pre-run benchmark.
