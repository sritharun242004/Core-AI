# Solution notes — mini-torch

## The graph is a recipe, not a tape of values

A `Tensor` stores `data`, a set of parent tensors, and a closure that knows how
to push an output gradient into those parents. `backward()` first builds a
reverse topological order so a shared intermediate is visited only after every
consumer has contributed its gradient. `_add_grad` accumulates rather than
assigning: this is what makes `z = x * x + x` correct.

## Broadcasting is part of the derivative

NumPy can add a `(features,)` bias to a `(batch, features)` matrix. Its backward
pass must sum the batch contributions back into the original bias shape. The
`_unbroadcast` helper removes prepended dimensions and sums dimensions whose
parent size was one. Ignoring this is a common bug: values look right, but a
bias receives only one row's gradient or has the wrong shape.

## Why stable cross-entropy subtracts a detached max

For logits `z`,

`log p_j = z_j - log(sum_k exp(z_k))`.

Subtracting `max(z)` inside the exponentials avoids overflow. The max is treated
as a constant in this teaching implementation. On regions where the maximum is
unique, that changes no derivative because the same shift cancels out of the
log-softmax; the resulting gradient is `softmax(z) - one_hot(target)`. A full
framework would also define a differentiable reduction for edge cases and use
fused kernels.

## Reduction is a contract

`none` returns one loss per example, `sum` adds examples, and `mean` divides by
batch size. For a mean cross-entropy loss, every row's gradient is scaled by
`1 / batch`; changing a reduction without adjusting a learning rate can make a
training curve appear mysteriously too slow or too fast. The tests compare all
three reductions and compare mean gradients with PyTorch when available.

## Why the synthetic fixture is honest

The fixture draws seven-segment glyphs into 28×28 arrays, adds seeded shifts and
noise, and returns the arrays directly. It is useful for checking that matrix
shapes, reductions, and a training loop work offline. It does **not** claim to
represent the data distribution or difficulty of the real MNIST benchmark. The
README and lesson call this out explicitly.

## Debugging checklist

- If a gradient has the right values but the wrong shape, inspect
  `_unbroadcast` and the matrix-multiply vector cases.
- If a loss is `nan`, inspect logits before exponentiation and keep the stable
  max shift in `cross_entropy`.
- If a model does not overfit six rows, first check `zero_grad`, then verify the
  target IDs and the reduction. Only after that tune the learning rate.
- If torch is unavailable, the parity tests are skipped; the finite-difference
  and shape/reduction tests remain local checks. A skip is not a claim of parity.

## Extensions for the challenge

Add `sigmoid`, a `softmax` helper, momentum SGD, or a tiny Adam optimizer. Keep
one reference numerical test for each new primitive and compare a small graph
against PyTorch before integrating it into the MLP.
