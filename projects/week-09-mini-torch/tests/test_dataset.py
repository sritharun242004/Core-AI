from __future__ import annotations

import numpy as np


def test_builtin_mnist_shaped_dataset_is_deterministic_and_offline() -> None:
    from mini_torch import make_mnist_shaped

    first = make_mnist_shaped(n_train=30, n_test=20, seed=123)
    second = make_mnist_shaped(n_train=30, n_test=20, seed=123)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)

    x_train, y_train, x_test, y_test = first
    assert x_train.shape == (30, 28 * 28)
    assert x_test.shape == (20, 28 * 28)
    assert y_train.shape == (30,)
    assert y_test.shape == (20,)
    assert x_train.min() >= 0.0 and x_train.max() <= 1.0
    assert set(y_train).issubset(set(range(10)))


def test_mnist_shaped_split_is_learnable() -> None:
    from mini_torch import MLP, accuracy, make_mnist_shaped, train_mlp

    x_train, y_train, x_test, y_test = make_mnist_shaped(
        n_train=100, n_test=40, noise=0.015, seed=5
    )
    model = MLP(28 * 28, hidden=(24,), out_features=10, seed=5)
    train_mlp(model, x_train, y_train, epochs=80, learning_rate=0.25, batch_size=25, seed=5)
    assert accuracy(model(x_train), y_train) >= 0.90
    assert accuracy(model(x_test), y_test) >= 0.75
