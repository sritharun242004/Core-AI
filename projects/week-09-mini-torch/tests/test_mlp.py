from __future__ import annotations

import numpy as np


def test_mlp_shapes_and_parameter_count() -> None:
    from mini_torch import MLP

    model = MLP(4, hidden=(8, 6), out_features=3, seed=7)
    logits = model(np.zeros((5, 4)))
    assert logits.shape == (5, 3)
    assert len(model.parameters()) == 6  # weight and bias for each of 3 Linear layers
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_mlp_overfits_tiny_three_class_table() -> None:
    from mini_torch import MLP, accuracy, train_mlp

    x = np.array(
        [[2.0, 0.0], [0.0, 2.0], [-2.0, -2.0], [1.5, 0.1], [0.1, 1.5], [-1.5, -1.0]],
        dtype=np.float64,
    )
    y = np.array([0, 1, 2, 0, 1, 2])
    model = MLP(2, hidden=(10, 10), out_features=3, seed=11)
    history = train_mlp(model, x, y, epochs=250, learning_rate=0.08)

    assert history["loss"][0] > history["loss"][-1]
    assert history["loss"][-1] < 0.08
    assert accuracy(model(x), y) == 1.0
