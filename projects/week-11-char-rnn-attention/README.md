# char-rnn-attention — Week 11

A small, inspectable PyTorch character-level sequence-to-sequence lab. An
LSTM (or GRU) encoder reads a character window, an LSTM/GRU decoder predicts
next characters, and Bahdanau-style additive attention learns which encoder states
to consult at each decoder step.

The default fixture is a tiny deterministic string held entirely in memory.
It is intentionally not Shakespeare and it never downloads data. That keeps
shape, normalization, gradient, and loss tests fast enough to run on every
laptop. A longer Shakespeare experiment is an optional exercise only after
you provide a local text file and record its provenance.

## Run the offline checks

From this directory:

```bash
PYTHONPATH=src ../../.venv/bin/python -m pytest
../../.venv/bin/ruff check src tests
```

If your checkout's virtual environment is at the repository root, the
following equivalent command can be used from the repository root:

```bash
PYTHONPATH=projects/week-11-char-rnn-attention/src \
  .venv/bin/python -m pytest projects/week-11-char-rnn-attention/tests
```

The notebook is percent-format Python and can be opened in Jupyter or
converted with Jupytext:

```bash
PYTHONPATH=src jupytext --to notebook notebooks/01-char-rnn-attention.py
```

## Public API

```python
from torch.utils.data import DataLoader

from char_rnn_attention import CharLSTMAttention, make_tiny_dataset
from char_rnn_attention.train import fit

dataset = make_tiny_dataset(seq_len=32, repeats=2)
model = CharLSTMAttention(
    dataset.vocab.size,
    embedding_size=32,
    hidden_size=48,
    attention_size=64,
)
history = fit(
    model,
    DataLoader(dataset, batch_size=32, shuffle=False),
    torch.optim.Adam(model.parameters(), lr=3e-3),
    epochs=3,
)
print(history["loss"])
print(model.generate(dataset.vocab, "re", max_new_tokens=40, temperature=0.0))
```

The core objects are:

- `CharVocab`: deterministic sorted character-to-index encoding and decoding.
- `TinyCharDataset` / `make_tiny_dataset`: overlapping next-character windows
  from an in-memory string. Every item is `(source_ids, target_ids)` with
  shape `(T,)`; target is source shifted one character.
- `AdditiveAttention`: learned `v(tanh(W_q q + W_h h))` scores and a softmax
  over source time. It accepts batched decoder queries and an optional padding
  mask, returning context vectors and normalized weights.
- `CharLSTMAttention`: LSTM encoder, teacher-forced LSTM decoder, additive
  attention, and a vocabulary projection. It returns logits `(B, T, V)` and
  attention weights `(B, T, S)`. Pass `rnn_type="gru"` for the same contract
  with GRU cells, or use `CharGRUAttention` directly.
- `sequence_cross_entropy`, `train_epoch`, `evaluate`, and `fit`: small
  token-weighted training utilities with optional gradient clipping.

`CharRNN` and `CharLSTM` are aliases for the concrete LSTM attention model;
`CharGRU` and `CharGRUAttention` expose the GRU variant. `BahdanauAttention`
is an alias for `AdditiveAttention`. A target sequence can be shorter or
longer than the source, and `teacher_forcing_ratio` supports fully forced,
free-running, or mixed decoding.

## What to inspect

1. Draw the `(B, source_steps, H)` encoder states and `(B, target_steps, H)`
   decoder queries. Attention produces `(B, target_steps, source_steps)`
   scores, so every target step gets a distribution over the source.
2. For one query, verify that `softmax(scores)` sums to one over source time.
   A mask must set padding positions to zero probability without changing the
   normalization of valid positions.
3. Compare an encoder-only recurrent model with the attention model on the
   same fixture. The point is not a benchmark; it is learning when a fixed
   final hidden state is an information bottleneck.
4. Watch token loss and gradient norms for a few deterministic steps before
   changing hidden size or learning rate. A loss that does not move usually
   indicates a target shift, shape, or mode error.

## Optional local Shakespeare experiment

The code does not fetch Shakespeare. To use a local file, read it explicitly,
construct `TinyCharDataset(text, seq_len=128)`, and write down the file's
source, character encoding, train/validation split, seed, and token count.
Keep that experiment separate from the offline tests; a score on a text file
is not comparable to the tiny fixture's plumbing checks.

## Honest boundaries

The tests establish a working recurrent forward pass, normalized additive
attention, finite gradients, token-wise cross-entropy reductions, and a
short loss-reduction smoke test. They do not establish Shakespeare quality,
long-context memory, or a production translation system. Teacher forcing can
make training look easier than free-running generation, so inspect generation
with a fixed prompt and report the decoding temperature and checkpoint.
