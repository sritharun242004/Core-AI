# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownLambdaType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportCallIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false
# %% [markdown]
# Week 13 — NanoGPT and an explicit SSM baseline
#
# This percent-format notebook is offline by default. It uses one deterministic
# in-memory character fixture and makes no network or pretrained-weight calls.

# %%
import torch
from nano_gpt_ssm import NanoGPT, TinySSMLanguageModel, make_tiny_dataset
from nano_gpt_ssm.train import fit, language_model_loss
from torch.utils.data import DataLoader

# %% [markdown]
# ## 1. The shifted character fixture
#
# A next-token target is the input shifted left by one character. Vocabulary
# ordering is sorted, so rerunning this cell gives the same ids.

# %%
torch.manual_seed(7)
dataset = make_tiny_dataset(seq_len=24, repeats=2)
x, y = dataset[0]
print("vocabulary:", dataset.vocab.size)
print("input:", dataset.vocab.decode(x.tolist()))
print("target:", dataset.vocab.decode(y.tolist()))
print("shapes:", x.shape, y.shape)

# %% [markdown]
# ## 2. Transformer anatomy and causal probabilities
#
# The first block exposes probabilities with shape `(batch, heads, query, key)`.
# Future key positions must be exactly zero, while each valid row sums to one.

# %%
model = NanoGPT(
    dataset.vocab.size,
    d_model=32,
    n_heads=4,
    n_layers=2,
    max_seq_len=24,
)
logits = model(x.unsqueeze(0))
probabilities = model.blocks[0].attention.attention_probabilities()
print("transformer logits:", logits.shape)
print("attention probabilities:", probabilities.shape)
print("first probability row:", probabilities[0, 0, 0])
print("row sum:", probabilities[0, 0].sum(-1))
print("future mass:", probabilities.triu(diagonal=1).abs().sum().item())

# %% [markdown]
# ## 3. Short fixed-batch training smoke test
#
# This loss curve validates connectivity and objective reduction; it is not a
# language-quality benchmark.

# %%
loader = DataLoader(dataset, batch_size=16, shuffle=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
history = fit(model, loader, optimizer, epochs=2)
print("NanoGPT losses:", history["loss"])

# %% [markdown]
# ## 4. The explicit state-space baseline
#
# TinySSM carries a state through a visible sequential recurrence. It uses the
# same vocabulary and next-token objective so the comparison has a shared
# plumbing contract, not a claim of equal architecture or speed.

# %%
ssm = TinySSMLanguageModel(dataset.vocab.size, embedding_size=20, state_size=32)
ssm_optimizer = torch.optim.AdamW(ssm.parameters(), lr=0.01, weight_decay=0.0)
ssm_history = fit(ssm, loader, ssm_optimizer, epochs=2)
print("TinySSM losses:", ssm_history["loss"])
print("final transformer loss:", float(language_model_loss(model(x[None]), y[None])))
print("final SSM loss:", float(language_model_loss(ssm(x[None]), y[None])))

# %% [markdown]
# ## 5. Causality check by perturbing the future
#
# Changing future ids cannot change earlier logits. This is the most useful
# qualitative check before discussing a loss curve.

# %%
changed = x.clone()
changed[16:] = (changed[16:] + 1) % dataset.vocab.size
with torch.no_grad():
    print(
        "transformer prefix unchanged:",
        torch.allclose(model(x[None])[:, :16], model(changed[None])[:, :16]),
    )
    print("SSM prefix unchanged:", torch.allclose(ssm(x[None])[:, :16], ssm(changed[None])[:, :16]))
