"""Autodiff engine tests. Includes a numeric parity check against torch.autograd
covering Review Focus #2 (gradient consistency)."""

from __future__ import annotations

import pytest


def test_forward_pass_arithmetic():
    from micrograd import Value

    a = Value(2.0)
    b = Value(-3.0)
    c = a * b + b.relu()
    assert c.data == pytest.approx(-6.0)


def test_backward_gradient_shapes():
    from micrograd import Value

    a = Value(2.0)
    b = Value(-3.0)
    c = a * b
    c.backward()
    assert a.grad == pytest.approx(-3.0)
    assert b.grad == pytest.approx(2.0)


def test_relu_gradient():
    from micrograd import Value

    a = Value(-1.0)
    b = a.relu()
    b.backward()
    assert b.data == 0.0
    assert a.grad == 0.0


def test_exp_and_log():
    from micrograd import Value

    x = Value(1.5)
    y = x.exp().log()
    y.backward()
    assert y.data == pytest.approx(1.5, rel=1e-6)
    assert x.grad == pytest.approx(1.0, rel=1e-6)


def test_matches_torch_autograd():
    """Review Focus #2 — numeric parity with torch.autograd on a non-trivial graph."""
    torch = pytest.importorskip("torch")
    from micrograd import Value

    def build_ours():
        a = Value(2.5)
        b = Value(-1.3)
        c = (a * b + a.relu()) * (b.exp() + Value(0.1))
        c.backward()
        return c.data, a.grad, b.grad

    def build_torch():
        a = torch.tensor(2.5, requires_grad=True)
        b = torch.tensor(-1.3, requires_grad=True)
        c = (a * b + a.relu()) * (b.exp() + 0.1)
        c.backward()
        return c.item(), a.grad.item(), b.grad.item()

    ours = build_ours()
    theirs = build_torch()
    assert ours[0] == pytest.approx(theirs[0], rel=1e-5)
    assert ours[1] == pytest.approx(theirs[1], rel=1e-5)
    assert ours[2] == pytest.approx(theirs[2], rel=1e-5)
