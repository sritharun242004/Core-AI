# Warmup — read one gradient graph (30 min)

1. Create `x = Tensor([[1.0, -2.0]], requires_grad=True)` and `w =
   Tensor([[0.5], [0.25]], requires_grad=True)`.
2. Compute `loss = ((x @ w).relu()).mean()` and write the derivative by hand
   before calling `loss.backward()`.
3. Add a two-element bias and confirm that its gradient sums contributions from
   every row in a batch.
4. Replace `mean()` with `sum()`. Explain why the gradient changes with batch
   size even though the forward predictions do not.

**Check:** include the graph's shapes and the hand calculation in your notes;
do not only report the final numbers.
