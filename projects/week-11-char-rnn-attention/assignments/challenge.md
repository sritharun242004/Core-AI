# Challenge — make attention an experiment (3+ hrs)

1. Add a padding-aware batch or variable-length collator. Write a failing test
   that padded source positions receive zero attention probability and that
   valid rows still sum to one.
2. Compare additive attention with an encoder-only baseline at a matched
   hidden width. Pre-register the prompt, seed, split, number of updates, and
   loss metric before looking at samples.
3. Add one controlled ablation: remove attention, change teacher-forcing ratio,
   clip gradients at a different norm, or increase sequence length. Keep all
   other settings fixed.
4. If using a local Shakespeare file, record its source and preprocessing and
   keep it out of git. Never infer a production system or company practice from
   a toy result.

**Deliverable:** an ablation note with a loss curve, at least two fixed-prompt
samples, attention visualisation or row summaries, and a failure analysis that
separates memory, optimization, and data effects.
