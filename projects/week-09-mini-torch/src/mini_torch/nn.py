"""Neural-network modules and a deterministic training loop."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .engine import Array, Tensor
from .losses import cross_entropy


class Linear:
    """An affine layer ``y = xW + b`` with visible Tensor parameters."""

    def __init__(self, in_features: int, out_features: int, *, rng: np.random.Generator) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Linear dimensions must be positive")
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            rng.normal(0.0, scale, size=(in_features, out_features)), requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, inputs: Tensor | object) -> Tensor:
        values = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        if values.ndim != 2 or values.shape[1] != self.weight.shape[0]:
            raise ValueError(
                f"expected a 2D batch with {self.weight.shape[0]} features, got {values.shape}"
            )
        return values @ self.weight + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


class MLP:
    """A stack of Linear layers with ReLU between hidden layers."""

    def __init__(
        self,
        in_features: int,
        hidden: Sequence[int] = (64, 32),
        out_features: int = 10,
        *,
        seed: int = 0,
    ) -> None:
        sizes = [in_features, *hidden, out_features]
        rng = np.random.default_rng(seed)
        self.layers = [
            Linear(sizes[index], sizes[index + 1], rng=rng) for index in range(len(sizes) - 1)
        ]

    def __call__(self, inputs: Tensor | object) -> Tensor:
        values = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        for index, layer in enumerate(self.layers):
            values = layer(values)
            if index < len(self.layers) - 1:
                values = values.relu()
        return values

    def parameters(self) -> list[Tensor]:
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()


class SGD:
    """The deliberately boring optimizer used in the lesson."""

    def __init__(self, parameters: Iterable[Tensor], learning_rate: float = 0.1) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.parameters = list(parameters)
        self.learning_rate = float(learning_rate)

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.data -= self.learning_rate * parameter.grad

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()


def accuracy(logits: Tensor | object, targets: object) -> float:
    """Classification accuracy from a batch of class logits."""

    values: Array = (
        logits.data if isinstance(logits, Tensor) else np.asarray(logits, dtype=np.float64)
    )
    labels: NDArray[np.int64] = np.asarray(targets, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != labels.size:
        raise ValueError("logits and targets must describe the same 2D batch")
    predictions: NDArray[np.int64] = np.argmax(values, axis=1)
    correct: NDArray[np.bool_] = predictions == labels
    return float(np.mean(correct))


def train_mlp(
    model: MLP,
    x: object,
    y: object,
    *,
    epochs: int = 20,
    learning_rate: float = 0.1,
    batch_size: int | None = None,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Train an MLP with deterministic shuffled mini-batches.

    The returned loss is a fresh full-dataset evaluation after every epoch,
    making curves comparable even when the optimizer uses mini-batches.
    """

    features = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("x must be (rows, features) and y must have one label per row")
    if epochs < 0:
        raise ValueError("epochs must be non-negative")
    if batch_size is None:
        batch_size = int(features.shape[0])
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    optimizer = SGD(model.parameters(), learning_rate=learning_rate)
    generator = np.random.default_rng(seed)
    history: dict[str, list[float]] = {"loss": [], "accuracy": []}
    for _ in range(epochs):
        order = generator.permutation(features.shape[0])
        for start in range(0, features.shape[0], batch_size):
            indices = order[start : start + batch_size]
            logits = model(Tensor(features[indices]))
            loss = cross_entropy(logits, labels[indices], reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        evaluation_logits = model(Tensor(features))
        evaluation_loss = cross_entropy(evaluation_logits, labels).item()
        history["loss"].append(evaluation_loss)
        history["accuracy"].append(accuracy(evaluation_logits, labels))
    return history


# A readable alias for callers who prefer the sklearn vocabulary.
fit = train_mlp


__all__ = ["MLP", "SGD", "Linear", "accuracy", "fit", "train_mlp"]
