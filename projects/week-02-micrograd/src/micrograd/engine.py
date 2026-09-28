"""Scalar autodiff engine — every op records its parents and a local gradient
function; backward() does a topo sort and applies the chain rule."""

from __future__ import annotations
import math
from typing import Callable, Iterable


class Value:
    __slots__ = ("data", "grad", "_prev", "_op", "_backward")

    def __init__(self, data: float, _children: Iterable["Value"] = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def _wrap(self, other: "Value | float") -> "Value":
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _b():
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _b
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _b():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _b
        return out

    def __pow__(self, other: float):
        assert isinstance(other, (int, float)), "power must be a plain number"
        out = Value(self.data ** other, (self,), f"**{other}")
        def _b():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _b
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _b():
            self.grad += (out.data > 0) * out.grad
        out._backward = _b
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _b():
            self.grad += out.data * out.grad
        out._backward = _b
        return out

    def log(self):
        assert self.data > 0, "log requires positive input"
        out = Value(math.log(self.data), (self,), "log")
        def _b():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _b
        return out

    def __neg__(self):        return self * -1
    def __sub__(self, other): return self + (-other if isinstance(other, Value) else Value(-other))
    def __radd__(self, other):return self + other
    def __rsub__(self, other):return (-self) + other
    def __rmul__(self, other):return self * other
    def __truediv__(self, other):
        other = self._wrap(other)
        return self * (other ** -1)
    def __rtruediv__(self, other):
        return self._wrap(other) * (self ** -1)

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[int] = set()
        def build(v: Value):
            if id(v) in visited: return
            visited.add(id(v))
            for c in v._prev: build(c)
            topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


__all__ = ["Value"]
