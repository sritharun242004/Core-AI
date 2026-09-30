# sft-lora-dpo-lab — Week 15a

An offline PyTorch post-training lab that keeps the policy small enough to
read and the post-training contracts visible:

1. a tiny decoder-only causal LM and deterministic supervised fine-tuning
   (SFT) fixture;
2. hand-written LoRA injection for named `nn.Linear` projections;
3. an optional symmetric int8 helper that explains the QLoRA boundary without
   requiring bitsandbytes, CUDA, a checkpoint, or a model download;
4. preference log-probability helpers for DPO, KTO, IPO, ORPO, and SimPO; and
5. checkpoint save/restore and a tiny deterministic training smoke test.

The DPO equation is attributed here to **Stanford's Rafailov et al. (2023),
Direct Preference Optimization: Your Language Model is Secretly a Reward
Model**. This project does not attribute DPO to Meta and does not claim to
reproduce the paper's model, data, or trainer.

## Offline checks

From this directory:

```bash
PYTHONPATH=src ../../.venv/bin/python -m pytest
../../.venv/bin/ruff check src tests notebooks
PYTHONPATH=src ../../.venv/bin/python notebooks/01-sft-lora-dpo.py
```

All data is generated in memory. The tests and notebook need no network,
credential, tokenizer package, pretrained weights, cloud SDK, or GPU. The
notebook is percent-format Python and can be opened in Jupyter or executed as
an ordinary script.

## Public API

```python
import torch
from post_training_lab import TinyCausalLM, inject_lora, sft_loss

policy = TinyCausalLM(vocab_size=16, d_model=16, n_heads=4, max_seq_len=8)
inject_lora(policy, rank=2, alpha=4, target_modules=("q_proj", "v_proj"))
ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
loss = sft_loss(policy(ids[:, :-1]), ids[:, 1:])
```

`LoRALinear` computes `base(x) + (alpha / rank) * B(A(x))`. The default
injection freezes dense/base parameters, clears stale base gradients, and
preserves existing adapters when adding another target. Matrix `B` starts at
zero, so adapter injection preserves the original logits until the first
adapter update. `mark_only_lora_trainable` identifies adapter parameters by
module identity rather than name substrings. LoRA checkpoints must be loaded
into the same injected rank/alpha/dropout architecture; loading validates the
state and restores `requires_grad` flags atomically.

`quantize_int8` returns a `QuantizedTensor(values, scale)` and
`dequantize_int8` restores it. This is a teaching representation, not NF4,
4-bit packing, an optimizer, or a benchmark. QLoRA in production combines a
quantized frozen base with trainable LoRA adapters and carefully chosen
compute/optimizer kernels; this lab only demonstrates the separation.

## Preference objectives

Each helper accepts already aggregated sequence log-probabilities so the
objective can be inspected without hiding a tokenizer or trainer. The caller
must state whether scores are sums or token means: `sequence_logprob` sums by
default and supports `normalize=True` for selected-token means.

- `dpo_loss`: reference-adjusted logistic margin; `dpo_batch_loss` shifts full
  causal sequences and evaluates the reference in temporary eval/no-grad mode;
- `kto_loss`: bounded logistic desirable/undesirable utility around a detached,
  non-negative batch KL approximation (not the paper's mismatched-pair
  estimator);
- `ipo_loss`: squared target-margin objective, normally on token means;
- `orpo_loss`: SFT anchor plus the true sequence-probability odds-ratio term;
  inputs must be mean log-probabilities; and
- `simpo_loss`: reference-free target-gap loss on token means, or sums plus
  explicit lengths.

The helpers are finite educational implementations of the named equations,
not claims that every production trainer uses these exact batching or KL
reductions.

## Evidence boundary

A lower toy loss proves that tensors, labels, and gradients are connected. It
does not establish instruction-following quality, harmlessness, factuality,
preference-model calibration, or generalization. LoRA reduces the number of
updated parameters; it does not guarantee equal quality or lower serving cost.
A preference objective inherits the quality and coverage of its data and
reference policy. Keep the local result separate from public company claims.
