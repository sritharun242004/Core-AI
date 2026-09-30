# %% [markdown]
# Week 14 — BPE, position, and a tiny causal LM
#
# Everything is in memory. The corpus is a teaching fixture, not the public
# TinyStories dataset.

# %%
from mini_bpe_pretrain import BPETokenizer, TinyCausalLM, TinyStoriesDataset, pretrain

# %%
corpus = [
    "the fox found a blue kite.",
    "the fox ran home and showed the kite to mia.",
    "mia smiled because the kite could fly high.",
]
tokenizer = BPETokenizer.train(corpus, vocab_size=48)
print("vocab:", tokenizer.vocab_size)
ids = tokenizer.encode("the fox", add_bos=True, add_eos=True)
print(ids, tokenizer.decode(ids))

# %%
dataset = TinyStoriesDataset(corpus, seq_len=16, tokenizer=tokenizer)
model = TinyCausalLM(tokenizer.vocab_size, d_model=24, n_heads=4, n_layers=1, max_seq_len=16)
result = pretrain(model, dataset, steps=8, batch_size=4, learning_rate=0.02, seed=3)
print("losses:", [round(value, 3) for value in result.losses])

# %%
inputs, _ = dataset[0]
print(
    "generated:",
    tokenizer.decode(
        model.generate(inputs[None], max_new_tokens=12)[0].tolist(), skip_special_tokens=True
    ),
)
print("parameters:", model.parameter_count())
