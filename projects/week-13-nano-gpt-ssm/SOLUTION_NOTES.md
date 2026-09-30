# Solution notes — causal attention and explicit state updates

## 1. The next-token contract

For a token sequence `x_0, ..., x_(T-1)`, the language model estimates
`p(x_t | x_<t)`. The fixture turns one character string into overlapping pairs:

```text
input  = text[i : i + T]
target = text[i + 1 : i + T + 1]
```

For batch size `B` and vocabulary size `V`, input and target ids are `(B, T)`;
the model returns logits `(B, T, V)`. Cross-entropy is applied independently
at each token and averaged. A uniform `V`-way prediction has loss `log(V)`.
That is a useful objective sanity check before training.

## 2. From a token to attention

A token id is mapped to an embedding `e_t ∈ R^d`. A fixed position vector
`p_t` is added so equal tokens at different positions need not be
indistinguishable:

```text
x_t = embedding(token_t) + p_t
p_t,2i   = sin(t / 10000^(2i/d))
p_t,2i+1 = cos(t / 10000^(2i/d))
```

For a head width `d_h`, learned projections form `Q = XW_Q`, `K = XW_K`, and
`V = XW_V`. Attention is a weighted read:

```text
scores = Q K^T / sqrt(d_h)
P      = softmax(scores over key positions)
head   = P V
```

The scale keeps dot products in a numerically useful range. The causal mask
sets score `(t, s)` to negative infinity when `s > t`. Therefore each row is a
distribution over the current and previous positions, and `P[t, s] = 0` for all
future `s`. The implementation stores probabilities before dropout for clear
testing and inspection.

Multiple heads split the model width, run this operation in parallel, then
concatenate and project. The code uses one fused `Linear(d, 3d)` for Q/K/V,
but the mathematical operation is unchanged.

## 3. The Transformer block

A block has two residual updates. The implementation is pre-normalized:

```text
u = x + Attention(LayerNorm(x))
y = u + MLP(LayerNorm(u))
MLP(z) = W_2 GELU(W_1 z + b_1) + b_2
```

The MLP is position-wise: the same two linear layers are applied to every time
position independently. Residual additions give each sublayer a short identity
path. LayerNorm normalizes features at one token position; it does not mix time
positions and hence cannot violate causality. The final normalized hidden state
is projected to vocabulary logits. Tying the LM-head weight to the token
embedding is a small parameter-sharing choice, not a requirement of GPT.

## 4. A minimal training step

For targets `y`,

```text
L = mean_t CrossEntropy(logits_t, y_t)
g = dL / dθ
θ <- AdamW(θ, g)
```

The training helper clears gradients, computes logits, reduces token loss,
backpropagates, optionally clips the gradient norm, and calls the optimizer.
The tests require finite, nonzero gradients and a substantial loss reduction on
a fixed batch after a short budget. This is a connectivity/sanity check, not a
held-out estimate.

## 5. Explicit state-space baseline

The comparison baseline keeps a state `s_t ∈ R^d` and scans left to right:

```text
s_t = A_t ⊙ s_(t-1) + B_t ⊙ x_t
z_t = C_t ⊙ s_t + D ⊙ x_t
```

`SelectiveSSM` predicts `A_t`, `B_t`, and `C_t` from `x_t`; sigmoid bounds the
decay and read coefficients, while the write coefficient uses `tanh`. The loop
is intentionally in Python so the recurrence, initial state, and chunked scan
are visible. `diagonal_ssm_scan` is the pure scan function used by the tests.

This resembles the broad state-space idea—compress history into a recurrent
state—but should not be called Mamba itself. It omits the particular continuous-
time discretization, parameterization, convolution/scan kernels, gating, and
systems work of named implementations. The model is useful for comparing a
causal fixed-state computation with rereading via attention, not for claiming
architecture parity.

## 6. Post-Transformer comparison map

- **Mamba:** a named selective state-space family. Public papers describe
  input-dependent state parameters and efficient scan implementations. This
  project borrows only the educational idea of an input-dependent recurrence.
- **Jamba:** a hybrid model family combining Transformer and state-space layers
  with other system/model choices. A tiny pure Transformer beside a toy SSM is
  not a Jamba reproduction.
- **Griffin:** a recurrent/hybrid family publicly described by Google research;
  its gated linear recurrence and local attention choices should be read from
  the cited release rather than inferred from this baseline.
- **This lab:** full causal attention in NanoGPT versus a diagonal, gated,
  input-dependent scan. Both use the same tokens and next-token loss. Report
  parameter count, sequence length, seed, optimizer, and token budget before
  interpreting curves.

A model being recurrent does not automatically mean it has linear wall-clock
cost: Python loops, memory movement, kernel fusion, batch shape, and hardware
matter. Conversely, attention's quadratic score matrix is a shape-level memory
statement, not a complete runtime benchmark.

## Checks established by the suite

The reference suite has 48 tests covering deterministic fixtures, target shift,
causal masks and future perturbations, attention normalization, a hand-computed
attention row, sinusoidal positions, MLP shapes, Transformer output and tied
weights, recurrence algebra and chunking, input-dependent coefficients, SSM
causality, finite gradients, loss reduction for both models, generation, and
invalid-shape contracts. No test downloads data or loads a checkpoint.
