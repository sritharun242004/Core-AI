"""Tiny NN layer on top of Value — Neurons, Layers, and a stacked MLP."""

from __future__ import annotations

import random

from .engine import Value


class Neuron:
    def __init__(self, nin: int, nonlin: bool = True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: list[Value]) -> Value:
        assert len(x) == len(self.w), f"input dim {len(x)} != weight dim {len(self.w)}"
        act = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), start=self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]


class Layer:
    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: list[Value]):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        sizes = [nin, *nouts]
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(nouts) - 1)) for i in range(len(nouts))
        ]

    def __call__(self, x: list[Value]):
        for layer in self.layers:
            x = layer(x)
            if not isinstance(x, list):
                x = [x]
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


__all__ = ["MLP", "Layer", "Neuron"]
