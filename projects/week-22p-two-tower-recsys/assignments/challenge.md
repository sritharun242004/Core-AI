# Challenge — 3+ hours

Choose one extension and retain the original tests:

- Add a content-feature item tower with a truly heldout new-item slice; compare with popularity and state which metadata was available at prediction time.
- Replace dense graph multiplication with sparse propagation. Prove equality on the tiny graph and measure memory/runtime at increasing graph sizes without claiming production scaling from a single point.
- Add candidate retrieval followed by a reranker. Report candidate recall, conditional ranking NDCG and full-pipeline NDCG separately; deliberately drop a relevant item to test denominators.

For any choice, run at least three seeds and report paired user-bootstrap intervals with explicit i.i.d.-user assumptions. Discuss exposure bias, feedback loops, privacy and why an offline gain is not an A/B treatment effect. Include an ablation and a stop/rollback criterion. Design guidance: `SOLUTION_NOTES.md`; this extension is not pre-implemented.
