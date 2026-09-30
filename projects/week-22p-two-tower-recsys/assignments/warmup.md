# Warmup — 30 minutes

1. Compute Recall@2 and NDCG@2 for ranking `[9,2]` with relevant items `{2,3,4}`. Explain the full-set denominators.
2. Repeat with no relevant items, k larger than the eligible catalog, and tied scores. State the API convention for each.
3. For BPR margin 0, derive loss and gradient. Which direction should one SGD update move the margin?
4. Draw a two-user/two-item bipartite graph with one heldout edge. Calculate one normalized neighbor coefficient using training degrees only.

Deliver a page of calculations and one regression test. Solutions: `SOLUTION_NOTES.md`.
