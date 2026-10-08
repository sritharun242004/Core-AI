"""Neuron / Layer / MLP tests — a tiny MLP should overfit 4 points."""

from __future__ import annotations

import random


def test_mlp_overfits_xor():
    from micrograd import Value, nn

    random.seed(1337)
    model = nn.MLP(2, [4, 4, 1])
    xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    ys = [0.0, 1.0, 1.0, 0.0]
    loss = Value(0.0)
    for _ in range(300):
        loss = Value(0.0)
        for x_row, y_target in zip(xs, ys, strict=True):
            pred = model([Value(x_row[0]), Value(x_row[1])])
            pred_val = pred[0]
            loss = loss + (pred_val - Value(y_target)) ** 2
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        for p in model.parameters():
            p.data -= 0.05 * p.grad
    assert loss.data < 0.5, f"loss too high: {loss.data}"
