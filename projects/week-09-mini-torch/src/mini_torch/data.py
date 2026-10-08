"""Offline, deterministic MNIST-shaped data for quick experiments.

The images are 28x28 grayscale arrays made from seven-segment digit templates
plus seeded sub-pixel shifts and noise. They are intentionally not a downloaded
copy of MNIST: the point is to test a learning loop without network access.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_SEGMENTS: dict[int, str] = {
    0: "abcdef",
    1: "bc",
    2: "abged",
    3: "abgcd",
    4: "fgbc",
    5: "afgcd",
    6: "afgecd",
    7: "abc",
    8: "abcdefg",
    9: "abfgcd",
}


def _template(label: int, shift_y: int, shift_x: int) -> FloatArray:
    image = np.zeros((28, 28), dtype=np.float64)
    y_top, y_mid, y_bottom = 4 + shift_y, 13 + shift_y, 22 + shift_y
    x_left, x_right = 5 + shift_x, 22 + shift_x
    segments: dict[str, tuple[slice, slice]] = {
        "a": (slice(y_top - 1, y_top + 2), slice(x_left, x_right + 1)),
        "g": (slice(y_mid - 1, y_mid + 2), slice(x_left, x_right + 1)),
        "d": (slice(y_bottom - 1, y_bottom + 2), slice(x_left, x_right + 1)),
        "f": (slice(y_top, y_mid + 1), slice(x_left - 1, x_left + 2)),
        "b": (slice(y_top, y_mid + 1), slice(x_right - 1, x_right + 2)),
        "e": (slice(y_mid, y_bottom + 1), slice(x_left - 1, x_left + 2)),
        "c": (slice(y_mid, y_bottom + 1), slice(x_right - 1, x_right + 2)),
    }
    for segment in _SEGMENTS[label]:
        rows, columns = segments[segment]
        image[rows, columns] = 1.0
    return image


def make_mnist_shaped(
    *,
    n_train: int = 500,
    n_test: int = 100,
    noise: float = 0.04,
    seed: int = 0,
) -> tuple[FloatArray, IntArray, FloatArray, IntArray]:
    """Return ``(x_train, y_train, x_test, y_test)`` with MNIST-like shapes.

    Class IDs are balanced by cycling through 0-9. A local NumPy generator is
    the only source of randomness, so repeated calls with the same arguments
    return byte-for-byte identical arrays and never access the network.
    """

    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    rng = np.random.default_rng(seed)

    def make_split(count: int) -> tuple[FloatArray, IntArray]:
        labels = np.arange(count, dtype=np.int64) % 10
        rows: list[FloatArray] = []
        for label in labels:
            shift_y, shift_x = rng.integers(-1, 2, size=2)
            image = _template(int(label), int(shift_y), int(shift_x))
            if noise:
                image = image + rng.normal(0.0, noise, image.shape)
            rows.append(np.clip(image, 0.0, 1.0).reshape(-1))
        return np.asarray(rows), labels

    x_train, y_train = make_split(n_train)
    x_test, y_test = make_split(n_test)
    return x_train, y_train, x_test, y_test


# Names used in exercises and notebooks; all point to the same offline fixture.
synthetic_mnist = make_mnist_shaped
load_mnist_shaped = make_mnist_shaped


__all__ = ["load_mnist_shaped", "make_mnist_shaped", "synthetic_mnist"]
