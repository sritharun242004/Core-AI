# nano-gpt-ssm — Week 13

An offline, from-scratch PyTorch language-model lab. The project puts two
small models beside one another on the same deterministic character fixture:

1. **NanoGPT**: token embeddings, fixed sinusoidal position encodings, causal
   multi-head self-attention, residual pre-norm blocks, position-wise GELU MLP,
   tied vocabulary head, and a next-token training step.
2. **TinySSM**: an explicit recurrent state-space baseline. `SelectiveSSM`
   makes diagonal decay/write/read coefficients from each input and scans
   `s_t = A_t * s_(t-1) + B_t * x_t`. It is deliberately a transparent
   teaching baseline, not a reproduction of Mamba, Griffin, or Jamba internals.

The fixture is a short in-memory sentence repeated twice. Vocabulary ordering,
windows, and tests are deterministic. No corpus, network request, pretrained
checkpoint, tokenizer package, accelerator, or external service is required.

## Offline checks

From this directory:

```bash
PYTHONPATH=src ../../.venv/bin/python -m pytest
../../.venv/bin/ruff check src tests
PYTHONPATH=src ../../.venv/bin/python notebooks/01-nano-gpt-ssm.py
```

The notebook is percent-format Python and can be opened in Jupyter or converted
with a local Jupytext installation. It prints shapes, causal probability
checks, losses, and the matched transformer/SSM comparison. All notebook work
stays in memory.

## Public API

```python
import torch
from nano_gpt_ssm import NanoGPT, TinySSMLanguageModel, make_tiny_dataset
from nano_gpt_ssm.train import language_model_loss, training_step

dataset = make_tiny_dataset(seq_len=24)
x, y = dataset[0]
model = NanoGPT(dataset.vocab.size, d_model=32, n_heads=4, n_layers=2, max_seq_len=24)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
loss = training_step(model, x[None], y[None], optimizer)
logits, probabilities = model(x[None]), model.blocks[0].attention.attention_probabilities()
```

`CausalSelfAttention(..., return_attention=True)` returns both the output and
`(batch, heads, query_time, key_time)` probabilities. Its lower-triangular
mask is a registered buffer, so future keys receive no probability. The MLP is
position-wise: it transforms each time position independently. The fixed
sinusoidal position code is the original Transformer-style `sin`/`cos` code,
not a learned embedding.

`diagonal_ssm_scan` exposes a hand-computable recurrence. `SelectiveSSM` keeps
the scan visible while allowing coefficients to depend on the current input.
`TinySSMLanguageModel` gives the SSM a vocabulary head so its loss can be
compared with NanoGPT under the same fixture and next-token objective.

## What to inspect

1. Hand-compute a two-token, one-head attention row: scale `QKᵀ` by
   `sqrt(head_dim)`, mask the future, and apply softmax over keys.
2. Change only future input ids and confirm the prefix logits do not change.
3. Compare transformer attention probabilities with the SSM's evolving state:
   attention rereads a bounded prefix at every layer; the SSM carries a fixed
   state through a sequential scan.
4. Run the same optimizer, seed, token budget, and fixture for both models.
   A lower toy loss is a plumbing observation, not an architecture benchmark.

## Honest boundary

The tests establish tensor shapes, causal masking, normalized probabilities,
finite gradients, a decreasing fixed-batch loss smoke test, and a recurrence
that agrees with hand calculations. They do not establish language quality,
long-context superiority, hardware efficiency, or parity with any named
production model. A production Mamba/SSM implementation involves additional
parameterizations, kernels, initialization choices, and engineering details;
this project intentionally does not claim to reproduce them.
