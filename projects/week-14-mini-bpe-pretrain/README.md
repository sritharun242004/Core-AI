# mini-bpe-pretrain — Week 14 reference project

Build a tiny, deterministic language-modeling stack without downloading a tokenizer, corpus, or checkpoint. The project has four layers:

1. a character-starting BPE tokenizer with deterministic merge ranks and special tokens;
2. RoPE and ALiBi positional helpers;
3. a small decoder-only causal LM with weight tying and causal attention;
4. an in-memory TinyStories-shaped fixture, short pretraining loop, and checkpoint round trip.

The fixture is deliberately synthetic and is **not** the TinyStories dataset. It exists so every test and notebook runs offline.

## Run

```bash
uv sync --extra dev
uv run pytest
```

## Public API

```python
from mini_bpe_pretrain import BPETokenizer, TinyCausalLM, TinyStoriesDataset, pretrain

corpus = ["a fox ran home.", "the fox found a kite."]
tokenizer = BPETokenizer.train(corpus, vocab_size=48)
dataset = TinyStoriesDataset(corpus, seq_len=16, tokenizer=tokenizer)
model = TinyCausalLM(tokenizer.vocab_size, d_model=32, n_heads=4, max_seq_len=16)
result = pretrain(model, dataset, steps=20, seed=7)
```

Use the implementation as a reference, not a production tokenizer. Real BPE systems need a byte-level pre-tokenizer, normalization policy, large corpora, reserved-token policy, and compatibility tests across versions.
