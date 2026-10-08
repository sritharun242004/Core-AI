# mini-torch — Week 9

A small, inspectable neural-network engine that extends the scalar ideas from
Week 2's `micrograd` to NumPy tensors. Every operation keeps its parents and a
local derivative closure; reverse-mode `Tensor.backward()` traverses that graph
in topological order. The project then uses the engine to train an MLP classifier
on a deterministic, built-in MNIST-shaped fixture.

This is educational code, not a replacement for PyTorch. The implementation
chooses a short graph and explicit arrays over a broad operator surface.

## Run

The project has no download step and does not need a network connection:

```bash
cd projects/week-09-mini-torch
PYTHONPATH=src ../../.venv/bin/python -m pytest
../../.venv/bin/ruff check src tests
```

The percent-format notebook can be run with Jupyter or converted with Jupytext:

```bash
PYTHONPATH=src jupytext --to notebook notebooks/01-mini-torch.py
```

## Public API

```python
import numpy as np
from mini_torch import MLP, accuracy, cross_entropy, make_mnist_shaped, train_mlp

x_train, y_train, x_test, y_test = make_mnist_shaped(n_train=500, n_test=100, seed=7)
model = MLP(28 * 28, hidden=(64, 32), out_features=10, seed=7)
history = train_mlp(
    model,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1,
    batch_size=50,
    seed=7,
)
print(history["loss"][-1], accuracy(model(x_test), y_test))
```

The core objects are:

- `Tensor`: NumPy data plus reverse-mode graph metadata; supports arithmetic,
  broadcasting, matrix multiplication, reductions, reshape, ReLU, `exp`, and
  `log`.
- `Linear` and `MLP`: parameter-owning layers with visible `Tensor` weights and
  biases.
- `cross_entropy` and `mse_loss`: stable losses with `none`, `sum`, and `mean`
  reduction semantics.
- `SGD` / `train_mlp`: a deliberately explicit optimizer and deterministic
  shuffled mini-batch loop.
- `make_mnist_shaped`: synthetic 28×28 seven-segment glyphs with seeded noise;
  it is MNIST-shaped, not the downloaded MNIST dataset.

## What to inspect

1. Read `src/mini_torch/engine.py` and trace one `matmul → add → relu → mean`
   graph forward and backward.
2. Compare the analytic gradient tests with `torch.autograd` when torch is
   installed. The parity test is optional and skips cleanly otherwise.
3. Change `reduction="mean"` to `"sum"` in a loss and observe how gradient
   magnitude scales with batch size.
4. Run the notebook and inspect the loss/accuracy history before changing the
   hidden-layer width.

## Constraints and honest boundaries

- The engine is CPU-only and uses float64 NumPy arrays for readable numerical
  parity. There is no GPU backend, graph compiler, checkpoint format, or
  production error handling.
- The fixture is designed for a fast training loop and shape sanity checks. It
  is not evidence of performance on real handwritten images.
- The implementation supports the operators used by this lesson, rather than
  pretending to be a complete tensor framework. Extending it is part of the
  challenge assignment.
