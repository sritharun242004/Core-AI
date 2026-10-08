"""A small, inspectable reverse-mode autodiff engine for NumPy arrays.

The implementation intentionally keeps the graph visible: every Tensor stores its
parents and a local ``_backward`` closure. ``backward`` topologically sorts that
graph, seeds the output, and applies the chain rule in reverse order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import EllipsisType
from typing import cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
type IndexPart = int | slice | EllipsisType | list[int] | NDArray[np.int_ | np.bool_] | None
type Index = IndexPart | tuple[IndexPart, ...]


def _unbroadcast(gradient: Array, shape: tuple[int, ...]) -> Array:
    """Undo NumPy broadcasting before adding a gradient to a parent."""

    result = np.asarray(gradient, dtype=np.float64)
    while result.ndim > len(shape):
        result = result.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and result.shape[axis] != 1:
            result = result.sum(axis=axis, keepdims=True)
    return result.reshape(shape)


class Tensor:
    """A NumPy value and the recipe needed to differentiate it."""

    __slots__ = ("_backward", "_op", "_prev", "data", "grad", "requires_grad")

    def __init__(
        self,
        data: object,
        requires_grad: bool = False,
        _children: Iterable[Tensor] = (),
        _op: str = "",
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: Array | None = (
            np.zeros_like(self.data, dtype=np.float64) if requires_grad else None
        )
        self.requires_grad = bool(requires_grad)
        self._prev = tuple(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def item(self) -> float:
        return float(self.data.item())

    def numpy(self) -> Array:
        return self.data.copy()

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float64)

    def _wrap(self, other: object) -> Tensor:
        return other if isinstance(other, Tensor) else Tensor(other)

    def _add_grad(self, gradient: Array) -> None:
        if self.requires_grad:
            value = np.asarray(gradient, dtype=np.float64)
            self.grad = value if self.grad is None else self.grad + value

    def __add__(self, other: object) -> Tensor:
        other_tensor = self._wrap(other)
        out = Tensor(
            self.data + other_tensor.data,
            self.requires_grad or other_tensor.requires_grad,
            (self, other_tensor),
            "+",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            self._add_grad(_unbroadcast(out.grad, self.shape))
            other_tensor._add_grad(_unbroadcast(out.grad, other_tensor.shape))

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: object) -> Tensor:
        return self + (-self._wrap(other))

    def __rsub__(self, other: object) -> Tensor:
        return self._wrap(other) - self

    def __mul__(self, other: object) -> Tensor:
        other_tensor = self._wrap(other)
        out = Tensor(
            self.data * other_tensor.data,
            self.requires_grad or other_tensor.requires_grad,
            (self, other_tensor),
            "*",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            self._add_grad(_unbroadcast(other_tensor.data * out.grad, self.shape))
            other_tensor._add_grad(_unbroadcast(self.data * out.grad, other_tensor.shape))

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> Tensor:
        return self * self._wrap(other) ** -1.0

    def __rtruediv__(self, other: object) -> Tensor:
        return self._wrap(other) / self

    def __pow__(self, exponent: object) -> Tensor:
        if not isinstance(exponent, (int, float)):
            raise TypeError("exponent must be a plain number")
        output_data = self.data**exponent
        out = Tensor(output_data, self.requires_grad, (self,), f"**{exponent}")

        def _backward() -> None:
            if out.grad is None:
                return
            self._add_grad(exponent * self.data ** (exponent - 1) * out.grad)

        out._backward = _backward
        return out

    def __matmul__(self, other: object) -> Tensor:
        other_tensor = self._wrap(other)
        out = Tensor(
            self.data @ other_tensor.data,
            self.requires_grad or other_tensor.requires_grad,
            (self, other_tensor),
            "@",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            gradient = out.grad
            if self.ndim == 1 and other_tensor.ndim == 2:
                grad_self = gradient @ other_tensor.data.T
                grad_other = np.outer(self.data, gradient)
            elif self.ndim == 2 and other_tensor.ndim == 1:
                grad_self = np.outer(gradient, other_tensor.data)
                grad_other = self.data.T @ gradient
            else:
                grad_self = gradient @ np.swapaxes(other_tensor.data, -1, -2)
                grad_other = np.swapaxes(self.data, -1, -2) @ gradient
            self._add_grad(_unbroadcast(grad_self, self.shape))
            other_tensor._add_grad(_unbroadcast(grad_other, other_tensor.shape))

        out._backward = _backward
        return out

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            self.requires_grad,
            (self,),
            "sum",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            gradient = out.grad
            if axis is None:
                expanded = np.broadcast_to(gradient, self.shape)
            else:
                axes = (axis,) if isinstance(axis, int) else axis
                normalized = tuple(a if a >= 0 else self.ndim + a for a in axes)
                expanded_gradient = gradient
                if not keepdims:
                    for a in sorted(normalized):
                        expanded_gradient = np.expand_dims(expanded_gradient, a)
                expanded = np.broadcast_to(expanded_gradient, self.shape)
            self._add_grad(expanded)

        out._backward = _backward
        return out

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        if axis is None:
            count = self.data.size
        elif isinstance(axis, int):
            count = self.shape[axis]
        else:
            count = int(np.prod([self.shape[a] for a in axis]))
        return self.sum(axis=axis, keepdims=keepdims) / float(count)

    def exp(self) -> Tensor:
        out = Tensor(np.exp(self.data), self.requires_grad, (self,), "exp")

        def _backward() -> None:
            if out.grad is None:
                return
            self._add_grad(out.data * out.grad)

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        if np.any(self.data <= 0):
            raise ValueError("log requires positive inputs")
        out = Tensor(np.log(self.data), self.requires_grad, (self,), "log")

        def _backward() -> None:
            if out.grad is None:
                return
            self._add_grad(out.grad / self.data)

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,), "relu")

        def _backward() -> None:
            if out.grad is None:
                return
            self._add_grad((self.data > 0.0) * out.grad)

        out._backward = _backward
        return out

    def reshape(self, *shape: int | tuple[int, ...]) -> Tensor:
        # NumPy validates invalid mixed tuple/scalar shapes itself; valid calls
        # supply either one tuple or individual integer dimensions.
        target_shape = (
            shape[0]
            if len(shape) == 1 and isinstance(shape[0], tuple)
            else cast(tuple[int, ...], shape)
        )
        out = Tensor(self.data.reshape(target_shape), self.requires_grad, (self,), "reshape")

        def _backward() -> None:
            if out.grad is None:
                return
            self._add_grad(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> Tensor:
        actual_axes = axes if axes else tuple(reversed(range(self.ndim)))
        out = Tensor(self.data.transpose(actual_axes), self.requires_grad, (self,), "transpose")
        inverse = np.argsort(actual_axes)

        def _backward() -> None:
            if out.grad is None:
                return
            self._add_grad(out.grad.transpose(inverse))

        out._backward = _backward
        return out

    @property
    def T(self) -> Tensor:  # noqa: N802 - NumPy-compatible transpose spelling
        return self.transpose()

    def __getitem__(self, index: Index) -> Tensor:
        out = Tensor(self.data[index], self.requires_grad, (self,), "slice")

        def _backward() -> None:
            if out.grad is None:
                return
            gradient = np.zeros_like(self.data)
            # ufunc.at stubs omit slices/tuples, which NumPy supports at runtime.
            cast(Callable[[Array, Index, Array], None], np.add.at)(gradient, index, out.grad)
            self._add_grad(gradient)

        out._backward = _backward
        return out

    def backward(self, gradient: object | None = None) -> None:
        """Accumulate derivatives from this tensor back to all graph leaves."""

        if gradient is None:
            if self.data.size != 1:
                raise ValueError("backward() needs a scalar output or an explicit gradient")
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(gradient, dtype=np.float64)
            if seed.shape != self.shape:
                raise ValueError(f"gradient shape {seed.shape} does not match {self.shape}")

        topology: list[Tensor] = []
        visited: set[int] = set()

        def visit(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._prev:
                visit(parent)
            topology.append(node)

        visit(self)
        self._add_grad(seed)
        for node in reversed(topology):
            node._backward()

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.data!r}, requires_grad={self.requires_grad})"


__all__ = ["Tensor"]
