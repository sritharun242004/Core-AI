# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# %% [markdown]
# Week 11 — character LSTM with additive attention
#
# This percent-format notebook is offline by default. The fixture is a tiny,
# deterministic in-memory text string; no corpus is downloaded.

# %%
import torch
from char_rnn_attention import CharLSTMAttention, make_tiny_dataset
from char_rnn_attention.train import fit
from torch.utils.data import DataLoader

# %% [markdown]
# ## 1. Character windows are a next-token contract
#
# Each item contains `x[t:t+T]` and `y[t+1:t+T+1]`. The vocabulary is sorted,
# so the same fixture always receives the same ids.

# %%
torch.manual_seed(7)
dataset = make_tiny_dataset(seq_len=24, repeats=2)
x, y = dataset[0]
print("vocabulary:", dataset.vocab.size)
print("source:", dataset.vocab.decode(x.tolist()))
print("target:", dataset.vocab.decode(y.tolist()))
print("window shapes:", x.shape, y.shape)

# %% [markdown]
# ## 2. Encoder, decoder, and additive attention
#
# The encoder returns one hidden state per source character. Each decoder query
# scores all source states with `v(tanh(Wq q + Wh h))`; softmax is over source
# time, so every attention row sums to one.

# %%
model = CharLSTMAttention(
    dataset.vocab.size,
    embedding_size=24,
    hidden_size=32,
    attention_size=40,
)
logits, weights = model(x.unsqueeze(0), y.unsqueeze(0))
print("logits:", logits.shape)
print("attention:", weights.shape)
print("first row sum:", weights[0, 0].sum().item())

# %% [markdown]
# ## 3. A short offline training smoke test
#
# The loss is a token mean. This validates the objective and gradients; it is
# not a Shakespeare benchmark.

# %%
loader = DataLoader(dataset, batch_size=32, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
history = fit(model, loader, optimizer, epochs=3)
print("loss history:", history["loss"])

# %% [markdown]
# ## 4. Free-running generation
#
# Teacher forcing supplied gold previous characters during training. Generation
# feeds the model's own prediction back in. Keep the prompt and temperature in
# an experiment log when comparing samples.

# %%
print(model.generate(dataset.vocab, "re", max_new_tokens=60, temperature=0.0))
