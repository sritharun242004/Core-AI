# Warmup — make the preference plumbing explicit

1. Run the tests and inspect `make_sft_examples` and
   `make_preference_pairs`. Draw the shapes of ids, labels, logits, and one
   sequence log-probability.
2. Hand-derive SFT cross-entropy for a two-token, three-vocabulary example.
   Explain what `-100` does and why input/label shifting must be stated.
3. For a frozen linear map with `d_in=16`, `d_out=32`, and rank `r=4`, count
   dense versus LoRA trainable parameters. Explain why zero-initialized `B`
   preserves initial logits.
4. Given policy chosen/rejected log-probs and a reference pair, compute the
   DPO margin and predict whether the loss increases when the rejected score
   rises. Name the paper and authors that introduced DPO in this lesson.
5. Compare DPO, KTO, IPO, ORPO, and SimPO by reference-policy use, data shape,
   and margin/anchor. Identify one thing a finite scalar loss cannot establish.

**Start with a failing test:** assert that masked sequence log-probabilities
ignore one token and that an equal policy/reference pair has the expected DPO
logistic value. Then run the isolated suite.
