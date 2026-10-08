# %% [markdown]
# Week 15a — SFT, LoRA, and preference objectives
#
# This percent-format notebook is offline by default. It uses fixed token ids,
# a tiny causal LM, and no tokenizer, checkpoint, API, or network path.

# %%
import copy
import tempfile
from pathlib import Path

import torch
from post_training_lab import (
    TinyCausalLM,
    dequantize_int8,
    dpo_batch_loss,
    inject_lora,
    ipo_loss,
    kto_loss,
    load_checkpoint,
    make_preference_pairs,
    make_sft_examples,
    mark_only_lora_trainable,
    orpo_loss,
    qlora_recipe,
    quantize_int8,
    save_checkpoint,
    sequence_logprob,
    simpo_loss,
    train_sft,
)

# %% [markdown]
# ## 1. Deterministic supervised fixture
#
# The labels are already aligned with each input position. A production data
# collator must document whether it shifts ids inside the model or here.

# %%
torch.manual_seed(15)
examples = make_sft_examples()
model = TinyCausalLM(vocab_size=16, d_model=16, n_heads=4, max_seq_len=8)
print("examples:", len(examples), "shape:", tuple(examples[0].input_ids.shape))
print("parameters:", model.num_parameters())
history = train_sft(model, examples, epochs=5, learning_rate=0.05)
print("SFT loss trace:", [round(value, 4) for value in history["loss"]])

# %% [markdown]
# ## 2. Inject a low-rank adapter
#
# Query/value projections are replaced by `LoRALinear`. B starts at zero, so
# the adapter is an identity-preserving change before its first update.

# %%
reference = copy.deepcopy(model).eval()
for parameter in reference.parameters():
    parameter.requires_grad = False
policy = copy.deepcopy(model)
inject_lora(policy, rank=2, alpha=4, target_modules=("q_proj", "v_proj"))
trainable = mark_only_lora_trainable(policy)
print("all parameters:", model.num_parameters())
print("LoRA trainable parameters:", trainable)
print("adapter recipe:", qlora_recipe())

# %% [markdown]
# ## 3. Optional int8 teaching round trip

# %%
weight = model.blocks[0].attn.q_proj.weight.detach()
packed = quantize_int8(weight)
restored_weight = dequantize_int8(packed)
print("int8 values:", tuple(packed.values.shape), packed.values.dtype)
print("max round-trip error:", float((restored_weight - weight).abs().max()))

# %% [markdown]
# ## 4. DPO and neighboring finite objectives
#
# Preference pairs contain categorical ids rather than natural-language text.
# The DPO attribution in this lab is Stanford's Rafailov et al. (2023).

# %%
pairs = make_preference_pairs()
chosen = torch.stack([pair.chosen_ids for pair in pairs])
rejected = torch.stack([pair.rejected_ids for pair in pairs])
dpo = dpo_batch_loss(policy, reference, chosen, rejected)
print("DPO batch loss:", float(dpo.detach()))
with torch.no_grad():
    # Full ids are shifted before scoring: logits[t] predicts ids[t + 1].
    chosen_scores = sequence_logprob(policy(chosen)[:, :-1], chosen[:, 1:], normalize=True)
    rejected_scores = sequence_logprob(policy(rejected)[:, :-1], rejected[:, 1:], normalize=True)
    reference_scores = sequence_logprob(reference(chosen)[:, :-1], chosen[:, 1:], normalize=True)
labels = torch.ones_like(chosen_scores, dtype=torch.bool)
print("KTO:", float(kto_loss(chosen_scores, reference_scores, labels)))
print("IPO:", float(ipo_loss(chosen_scores, rejected_scores)))
print("ORPO:", float(orpo_loss(chosen_scores, rejected_scores)))
print("SimPO:", float(simpo_loss(chosen_scores, rejected_scores)))

# %% [markdown]
# ## 5. Checkpoint round trip

# %%
with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "toy-policy.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    save_checkpoint(model, optimizer, checkpoint, step=5, loss=history["loss"][-1])
    restored = TinyCausalLM(vocab_size=16, d_model=16, n_heads=4, max_seq_len=8)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.05)
    metadata = load_checkpoint(restored, restored_optimizer, checkpoint)
    identical = all(
        torch.equal(left, right)
        for left, right in zip(model.parameters(), restored.parameters(), strict=True)
    )
    print("checkpoint metadata:", metadata)
    print("exact model restore:", identical)
