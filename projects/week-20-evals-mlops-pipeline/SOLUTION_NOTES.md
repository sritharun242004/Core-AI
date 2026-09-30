# Solution notes

## Release decision

For paired case scores, compute d_i = candidate_i - baseline_i. Estimate the mean delta and bootstrap entire case differences; independently resampling each model discards useful pairing. The conservative gate requires the lower 95% percentile endpoint >= -max_drop. Small samples, dependent prompts and repeated threshold tuning can invalidate the apparent confidence. A gate is an engineering policy, not proof of safety.

## Judge failure exercise

A judge that always selects the left answer yields mapped scores 1 and 0 after a swap: average .5, consistency false. A genuine tie produces .5 twice, consistency true. Validate the output schema before counting a vote. Real model judges can be prompt-injected by candidate text and biased toward verbosity; require human adjudication on disagreements and a labeled calibration set.

## Drift and leakage

Fit histogram boundaries on training/reference values, not the current production batch. Include infinite tail boundaries so shifted points are not discarded. Add small pseudocounts then renormalize. Identical empirical histograms give zero PSI. PSI cannot detect label-conditional changes from feature marginals alone. Constant features need separate change-of-value checks in production.

ID separation is necessary but insufficient: entities, temporal windows, near duplicates and generated paraphrases can cross splits. The n-gram index deliberately excludes the same ID; the split contract separately forbids overlap. A shared question template creates overlap without answer leakage. Investigate rather than announcing contamination from one threshold.

## Worked numeric example

Baseline scores [0,1,1,0], candidate [1,1,0,1] produce differences [1,0,-1,1] and mean improvement .25. Four cases are far too few for a confident deployment decision. Write the result and uncertainty, add cases without peeking at their outcomes, then make a predeclared decision.

## Interview rubric

Baseline: distinguish train/validation/test and choose task metrics. Senior: paired intervals, versioned scorer/data/model, judgment calibration, failure slices. Staff: gate ownership, shadow/canary rollout, rollback, delayed-label monitoring, privacy/retention and budget. No benchmark score substitutes for product-specific acceptance.
