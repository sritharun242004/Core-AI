# Solution notes — Week 15b

These notes solve the local exercise, not the public DeepSeek papers.

## Warmup

For `N` flattened tokens, `E` experts, and top-k `K`, router logits and
all-expert probabilities are `(N,E)`. Selected indices and selected weights are
`(N,K)`. For `K > 1`, the selected softmax runs over `K` so each token's dispatched
mixture has total weight one. Top-1 retains the selected all-expert probability
instead of applying a one-element softmax, so the task loss can train the
router. With `N=12`, `E=4`, `K=2`, and capacity factor `1`,
capacity is `ceil(12*2/4)=6` slots per expert. A hot expert can request more
than six assignments; the dispatch mask makes the overflow explicit.

The toy balance term multiplies mean all-expert router probability by the
fraction of selected assignments. It is finite even though top-k membership is
discrete. The auxiliary loss is a pressure toward utilization, not an
interpretation of what an expert has learned.

For rewards `[1, 1, 1, 1]`, centered and scaled advantages are all zero. For
`[0, 0, 0, 1]`, the correct candidate is positive and the three others are
negative; their group mean is zero after centering.

## Build observations

The synthetic classifier should reduce cross-entropy plus a small balance term
on a fixed two-dimensional separator. At a low capacity factor, selected loads
can exceed dispatched loads and some token branches become zero. Increasing
capacity reduces drops but can increase expert work. A report should include
both counters rather than selecting the one that tells the preferred story.

The parser accepts `What is 17 + 25?` and rejects arbitrary embedded syntax. A
response such as `17 + 25 = 42` is accepted through its explicit final number;
`4/1`, `4 or 5`, and a response with no final numeric answer are rejected. This
is a narrow deterministic outcome verifier, not a general expression parser.

`group_relative_advantages` centers and scales rewards per prompt and returns
zeros for a constant group. GRPO was introduced in DeepSeekMath (Shao et al.,
2024, arXiv:2402.03300), and DeepSeek-R1 (arXiv:2501.12948, 2025) later used
it for its reasoning RL; R1 did not invent the method. The local `grpo_loss`
uses a detached old-policy snapshot,
ratio clipping, and optional squared reference-log-probability penalty. The
lab's answer-ID policy update is intentionally only GRPO-like: it omits token
masks, variable-length sequence aggregation, distributed rollout, and the
multi-stage R1 data/training pipeline.

A majority vote spends one verifier/model call per sample. It helps a seeded
noisy sampler when correct answers are more frequent, but the local result does
not generalize to DeepSeek-R1, GSM8K, or open-ended reasoning.
