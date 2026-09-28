# Warmup — trace one recurrent step (30–45 min)

1. Write the shapes of an embedding `x_t`, hidden state `h_t`, and cell state
   `c_t` for batch size 2, embedding width 8, and hidden width 12.
2. Expand the four LSTM gate equations and label the input, forget, candidate,
   and output paths.
3. Construct `AdditiveAttention(hidden_size=12)` and verify that a query batch
   of shape `(2, 4, 12)` over values `(2, 7, 12)` returns context `(2, 4, 12)`
   and weights `(2, 4, 7)`.
4. Start with a failing assertion that each attention row sums to one, then
   implement or repair the softmax axis.

**Deliverable:** a one-page shape table plus a paragraph explaining why the
cell-state addition is easier to optimize than repeatedly multiplying a fresh
recurrent matrix.
