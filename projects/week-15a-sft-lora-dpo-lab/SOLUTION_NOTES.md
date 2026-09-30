# Solution notes — Week 15a

These notes solve the local exercise. They are not a claim about any private
training stack or a reproduction of a published model.

## Warmup

For input ids `x_0, ..., x_{T-1}`, a causal LM returns logits
`z_t = f(x_{0:t})`. Cross-entropy compares `z_t` with the target token at the
same position in this fixture. A conventional shifted implementation instead
feeds `x_{0:T-2}` and compares against `x_{1:T-1}`; both conventions are
valid if the input/label alignment is explicit. `sft_loss` uses direct
`[batch, time]` labels and ignores `-100` positions.

The per-sequence log-probability is:

```text
log p(y | x) = sum_t mask_t * log_softmax(logits_t)[y_t]
```

`sequence_logprob` expects already aligned logits/labels, ignores `-100` even
when a mask is supplied, and can divide by the selected-token count with
`normalize=True`. `dpo_batch_loss` owns the causal shift (`logits[:, :-1]`
against `ids[:, 1:]`) before calling that helper. Summing versus averaging is
an objective choice: DPO may use sums, while IPO/ORPO/SimPO commonly use
length-normalized completion scores.


## LoRA accounting

For a frozen linear layer `W` with `d_in` inputs and `d_out` outputs, LoRA
adds `A in R^(r x d_in)` and `B in R^(d_out x r)`:

```text
h = W x + (alpha / r) B A x
```

The adapter has `r(d_in + d_out)` trainable scalars versus `d_out*d_in` in
the dense map. Zero-initializing `B` gives an identity-preserving injection.
The local implementation targets leaf names such as `q_proj` and `v_proj` and
freezes every non-adapter parameter. It does not merge, shard, or quantize
optimizer state.

Quantizing a tensor symmetrically uses

```text
scale = max(abs(W)) / 127
q = clamp(round(W / scale), -127, 127)
W_hat = q * scale
```

The error is bounded by roughly half a scale per value, subject to clipping.
The helper's per-tensor scale is intentionally simple. QLoRA normally means a
quantized frozen base plus LoRA updates, often with a specialized 4-bit
format and paged optimizer; the local int8 round trip is only an explanation.
DoRA adds a magnitude component to a directional low-rank update. It is not
implemented here, because adding an untested parameterization would hide the
lesson's main contracts.

## Preference objectives

**DPO (Stanford Rafailov et al., 2023)** compares a policy and frozen
reference through a preference margin:

```text
m = (log pi(y+) - log pi(y-)) - (log pi_ref(y+) - log pi_ref(y-))
L_DPO = -log sigmoid(beta * m)
```

The project explicitly attributes this method to Stanford's Rafailov et al.
2023 paper, not to Meta. The data fixture supplies chosen/rejected token ids,
but does not train a reward model.

The other helpers make neighboring families comparable without claiming exact
trainer fidelity:

- IPO squares the difference between a reference-adjusted margin and
  `1/(2 beta)`.
- KTO receives independent desirable/undesirable examples and applies the
  bounded `1 - sigmoid` utility around a detached, non-negative batch KL
  approximation. This fixture reuses the labeled batch rather than estimating
  the paper's mismatched-pair reference point.
- ORPO adds an SFT anchor to `-log sigmoid(log odds(chosen) - log
  odds(rejected))`; raw log-probability differences are not odds ratios.
- SimPO removes the reference policy and applies a target margin `gamma` to
  length-normalized chosen versus rejected scores.

For all five, finite values and gradient flow are necessary but not sufficient
for a good post-training method. A real run must specify sequence masks,
length normalization, reference checkpoint, beta/gamma, data filtering,
preference noise, validation prompts, and safety evaluations.

## Training and checkpoints

`train_sft` uses AdamW with no weight decay on a fixed in-memory fixture. The
expected observation is a decreasing cross-entropy trace, not a language
benchmark. `save_checkpoint` stores CPU model tensors, optional optimizer
state, an integer step, and caller metadata. A restored model must match every
state tensor exactly before continuing training. Checkpoint portability is not
checkpoint safety: load only files from a trusted source in a real project.

## What the local test can establish

- causal logits have the expected `[batch, time, vocab]` shape;
- ignored labels do not contribute to SFT loss;
- masked sequence log-probabilities have the expected sum;
- LoRA adapters preserve initial logits, freeze/clear dense base gradients,
  retain existing adapters when targets are added, and reduce trainable count;
- int8 pack/unpack error is finite and small on the fixture;
- DPO/KTO/IPO/ORPO/SimPO return finite scalar losses;
- a deterministic fixture produces a decreasing SFT trace; and
- a checkpoint restores model and optimizer state.

It cannot establish instruction-following, harmlessness, reward-model validity,
RLHF/RLAIF quality, a production QLoRA speedup, or any company's current
architecture. Those require named public evidence and a separately designed
evaluation.
