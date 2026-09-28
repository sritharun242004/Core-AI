# Solution notes — recurrent models and additive attention

## 1. Start with the recurrent state

A vanilla recurrent network reads one input vector at a time and carries a
state forward:

`h_t = tanh(W_x x_t + W_h h_{t-1} + b_h)`

A character embedding turns a token id into `x_t`. The same `W_x`, `W_h`, and
`b_h` are reused at every time step, which is why the model can process a
sequence longer than the one used during initialization. The output layer maps
the state to vocabulary logits:

`p(y_t | x_<=t) = softmax(W_o h_t + b_o)`.

Unrolling the recurrence makes the optimization trade-off visible: it shares
parameters across time, but a gradient must pass through many copies of the
state transition. A saturating nonlinearity can make that product very small
or very large.

## 2. LSTM gates create controlled memory paths

An LSTM separates a cell state `c_t` from the exposed hidden state `h_t`. With
`[h_{t-1}, x_t]` as the concatenated input, its gates are:

```text
i_t = sigmoid(W_i [h_{t-1}, x_t] + b_i)   # write/input gate
f_t = sigmoid(W_f [h_{t-1}, x_t] + b_f)   # forget gate
o_t = sigmoid(W_o [h_{t-1}, x_t] + b_o)   # expose/output gate
g_t = tanh   (W_g [h_{t-1}, x_t] + b_g)   # candidate content
c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

The additive cell update gives the derivative a route scaled by `f_t` rather
than multiplying a fresh recurrent matrix at every step. This is a useful
inductive bias, not a guarantee against exploding gradients or forgetting.
Gradient clipping in the tiny training helper is an explicit safety valve,
not a substitute for checking the data and objective.

## 3. Seq2seq separates source and target time

For a source sequence `x_1:S` and target `y_1:T`, an encoder computes states
`h_1:S`. A decoder models the factorization:

`p(y_1:T | x_1:S) = product_t p(y_t | y_<t, x_1:S)`.

At training time, teacher forcing supplies the previous gold target to the
decoder. The first source character serves as a small deterministic BOS token
in this reference implementation; a production tokenizer would normally add
an explicit BOS/EOS vocabulary entry. At generation time, the decoder instead
feeds back its own previous prediction. This train/free-run difference is why a
falling teacher-forced loss is not enough evidence of useful generation.

The project uses equal source and target lengths so a language-model window is
simple: `x = text[t:t+T]`, `y = text[t+1:t+T+1]`. The model still has the
seq2seq shape contract, and its additive attention distribution is visible for
every prediction step.

## 4. Additive attention is a learned soft lookup

A fixed final encoder state is a bottleneck: it must summarize every source
character before the decoder sees any of them. Additive attention lets each
decoder query choose a weighted combination of all encoder states:

`e_{t,s} = v_a^T tanh(W_q q_t + W_h h_s)`

`alpha_{t,:} = softmax(e_{t,:})`

`context_t = sum_s alpha_{t,s} h_s`.

Here `q_t` is the decoder state and `h_s` is an encoder state. The softmax is
along source time `s`, therefore `sum_s alpha_{t,s} = 1` for every target step
`t`. A padding mask sets invalid energies to a large negative value before
softmax; valid positions remain normalized. This is *additive* attention: the
query and key are projected, added, passed through `tanh`, and scored. It is
not scaled dot-product attention, which arrives in Week 13's transformer.

The output head receives both the decoder state and the selected context:

`logits_t = W [q_t ; context_t] + b`.

Concatenating both paths lets the decoder retain local recurrent state while
using the retrieved source evidence.

## 5. Loss and what the tests establish

For vocabulary size `V`, target id `y_t`, and logits `z_t`, token loss is:

`ell_t = -log softmax(z_t)_{y_t}`.

A batch mean is the mean over `B*T` tokens, while a sum scales with the number
of tokens. `sequence_cross_entropy` exposes `none`, `mean`, and `sum` so the
reduction is testable rather than hidden in a training loop. The tests cover:

- deterministic fixture windows and round-trip vocabulary encoding;
- model output shapes `(B,T,V)` and `(B,T,S)`;
- attention weights that are non-negative and sum to one over source time;
- masked positions receiving exactly zero probability;
- finite gradients through embedding, recurrent, attention, and output layers;
- the identity `sum(none) == sum` and `mean(none) == mean`;
- a short CPU run whose token loss falls without a download.

Those are implementation contracts. They are not evidence that this tiny model
can reproduce Shakespeare, translate a language, or match a company's private
sequence architecture.

## Debugging checklist

- If `logits.shape` is `(B,T,V)` but targets are `(B,T+1)`, fix the window
  shift before touching the optimizer.
- If attention rows do not sum to one, inspect the softmax dimension; it must be
  source time, not batch or hidden size.
- If masked weights become NaN, check whether a sample masks every source
  position. A valid sequence needs at least one unmasked state.
- If loss is constant, inspect `requires_grad`, `optimizer.zero_grad`, target
  ids, and whether the model was accidentally left in evaluation mode.
- If free-running output collapses to one character, compare teacher-forced and
  autoregressive decoding, lower temperature only after checking the logits,
  and report the prompt and checkpoint rather than cherry-picking a sample.
