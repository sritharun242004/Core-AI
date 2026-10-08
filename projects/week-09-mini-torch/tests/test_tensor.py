# pyright: reportCallIssue=false, reportArgumentType=false
"""Tensor/autograd contract: write these before the implementation."""

from __future__ import annotations

import numpy as np
import pytest


def test_matmul_relu_mean_backprop_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    from mini_torch import Tensor

    x_value = np.array([[0.2, -0.4], [1.1, 0.7]], dtype=np.float64)
    w_value = np.array([[0.3, -0.8, 0.2], [0.5, 0.6, -0.1]], dtype=np.float64)
    b_value = np.array([0.1, -0.2, 0.05], dtype=np.float64)

    x = Tensor(x_value, requires_grad=True)
    w = Tensor(w_value, requires_grad=True)
    b = Tensor(b_value, requires_grad=True)
    loss = ((x @ w + b).relu() ** 2).mean()
    loss.backward()

    tx = torch.tensor(x_value, requires_grad=True)
    tw = torch.tensor(w_value, requires_grad=True)
    tb = torch.tensor(b_value, requires_grad=True)
    torch_loss = ((tx @ tw + tb).relu() ** 2).mean()
    torch_loss.backward()

    assert loss.item() == pytest.approx(torch_loss.item(), rel=1e-10)
    np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(w.grad, tw.grad.numpy(), rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(b.grad, tb.grad.numpy(), rtol=1e-8, atol=1e-9)


def test_broadcasting_and_repeated_use_accumulate_gradients() -> None:
    from mini_torch import Tensor

    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    bias = Tensor([0.5, -1.0], requires_grad=True)
    output = (x + bias) * (x + bias)
    output.sum().backward()

    np.testing.assert_allclose(x.grad, [[3.0, 2.0], [7.0, 6.0]])
    np.testing.assert_allclose(bias.grad, [10.0, 8.0])


def test_backward_requires_scalar_or_explicit_gradient() -> None:
    from mini_torch import Tensor

    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError, match="scalar"):
        x.backward()
    x.backward(np.array([2.0, 3.0]))
    np.testing.assert_allclose(x.grad, [2.0, 3.0])


def test_zero_grad_resets_array_in_place() -> None:
    from mini_torch import Tensor

    x = Tensor(2.0, requires_grad=True)
    (x * x).backward()
    x.zero_grad()
    np.testing.assert_allclose(x.grad, 0.0)
