# moe-reasoning-lab — Week 15b

An offline PyTorch lab that puts two ideas beside one another:

1. **Sparse mixture of experts (MoE):** a learned router selects top-k
   position-wise experts, normalizes selected weights for k > 1 (and retains
   the selected router probability for trainable top-1 gating), enforces an
   explicit per-expert capacity, and reports a load-balancing auxiliary loss.
2. **Verifiable toy reasoning:** deterministic arithmetic questions provide a
   reward/verifier, grouped relative advantages, a small GRPO-like policy
   update, and test-time sampling with majority vote.

The code is intentionally small enough to read in one sitting. It has no
corpus, checkpoint, tokenizer, API key, cloud SDK, or network path. The toy
loss curves validate tensor plumbing and optimization only; they are not
language-quality, reasoning, scaling, or production-throughput benchmarks.

## Offline checks

From this directory:

```bash
PYTHONPATH=src ../../.venv/bin/python -m pytest
../../.venv/bin/ruff check src tests
PYTHONPATH=src ../../.venv/bin/python notebooks/01-moe-reasoning.py
```

The notebook is percent-format Python and can be opened in Jupyter or run as a
script. It prints routing shapes, capacity/load statistics, a supervised MoE
loss trace, verifier rewards, group advantages, one policy loss, and a
majority-vote result. No Astro build or network download is needed.

## Public API

```python
import torch
from moe_reasoning_lab import TinyMoE, verify_arithmetic, group_relative_advantages

moe = TinyMoE(d_model=16, hidden_dim=32, num_experts=4, top_k=2)
x = torch.randn(2, 8, 16)
y, aux_loss = moe(x, return_aux=True)
routing = moe.last_routing
assert y.shape == x.shape
assert torch.allclose(routing.topk_weights.sum(-1), torch.ones(16))
assert verify_arithmetic("What is 17 + 25?", "42")
advantages = group_relative_advantages(torch.tensor([[0.0, 1.0, 0.0, 1.0]]))
```

`TopKRouter` exposes `probabilities`, `topk_indices`, `topk_weights`,
`dispatch_mask`, `expert_loads`, `selected_loads`, `capacity`, and `aux_loss`
in a `Routing` record. Capacity is `ceil(capacity_factor * tokens * top_k /
experts)` and assignments beyond that deterministic budget are marked dropped.
`selected_loads` is the pre-capacity integer count used by the auxiliary loss,
while `expert_loads` is the successfully dispatched integer count. For k > 1,
selected weights sum to one before dropping; top-1 retains the selected
all-expert probability so the task loss has a router gradient. This makes the
distinction between router balance, task trainability, and actual capacity
visible.

## Model and reasoning boundaries

- The expert is a two-layer GELU MLP, not a full Transformer block.
- The auxiliary term is a Switch-style importance/load proxy. It is a
  teaching diagnostic, not a claim to reproduce a particular vendor's router.
- `verify_arithmetic` parses only simple binary `+`, `-`, `*`, and `/`
  expressions and checks the final numeric answer. It is not a general theorem
  prover or a safe evaluator for arbitrary text.
- `grpo_loss` demonstrates within-prompt relative rewards, a detached
  advantage, optional ratio clipping, and an optional reference penalty. It is
  deliberately not presented as a reproduction of all DeepSeek R1 training
  details.
- Majority vote can improve a noisy deterministic sampler's answer only when
  correct candidates are common. It adds test-time compute and does not make
  an uninformative verifier reliable.
