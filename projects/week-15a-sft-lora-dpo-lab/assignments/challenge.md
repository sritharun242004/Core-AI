# Challenge — make post-training claims falsifiable

Choose one extension and add tests first:

- Add prompt/completion masks so DPO scores only completion tokens, then
  compare sum versus length-normalized log-probabilities on short and long
  responses.
- Implement a tiny preference update over the fixture and compare DPO with
  IPO, ORPO, and SimPO under matched seeds, update counts, and reference
  checkpoints. Report chosen win rate, KL/reference gap, and held-out prompts.
- Add a trainable DoRA-style magnitude vector beside the LoRA direction. Test
  identity initialization and parameter counts; explain what remains unlike a
  production PEFT library.
- Add a local adapter merge/export function and test merged versus unmerged
  logits at tolerance. Include dtype, rank, alpha, and checkpoint metadata.
- Add synthetic preference noise and a held-out template split. Measure how
  noisy labels affect each finite objective rather than selecting the best
  curve.

Every extension must stay offline and include seed, device, model dimensions,
sequence mask, objective hyperparameters, reference policy, update budget,
metrics, and failure cases. Do not call a toy loss an alignment result, do not
claim DPO is a Meta method, and do not infer private company internals from a
public paper or local adapter experiment.
