# Build — SFT, adapters, and one preference update

1. Run `notebooks/01-sft-lora-dpo.py` offline. Record Python/PyTorch version,
   seed, device, vocabulary size, sequence length, model parameters, and the
   SFT loss trace.
2. Inject rank-2 LoRA into query/value projections. Compare total and
   trainable parameters, verify the initial logits are unchanged, then train
   only adapter parameters on the fixed fixture.
3. Run the int8 round trip on at least three tensors, report maximum absolute
   error, and explain why this helper is not NF4 QLoRA. Return the recipe
   explanation from `qlora_recipe` in your notes.
4. Make a frozen reference copy of the policy, compute DPO on deterministic
   chosen/rejected sequences, and inspect policy/reference sequence log-probs.
   Evaluate the KTO, IPO, ORPO, and SimPO helpers on the same margins.
5. Save and restore a checkpoint. Compare every state tensor, optimizer step,
   metadata, and a post-restore loss. Do not use a downloaded checkpoint or
   remote dataset.

**Acceptance checks:** isolated pytest, Ruff, and notebook pass; the SFT trace
reduces; adapter injection is identity-preserving; every objective is finite;
and restore is exact. Report the difference between these plumbing checks and
instruction-following or safety evidence.
