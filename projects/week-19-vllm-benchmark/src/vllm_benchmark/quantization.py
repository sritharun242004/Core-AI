"""Symmetric, per-last-axis-group quantization, not GPTQ/AWQ or packed kernels."""

from dataclasses import dataclass

import numpy as np


def positive_int(value: int, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be an integer >= {0 if allow_zero else 1}")


@dataclass(frozen=True)
class QuantizedWeights:
    codes: np.ndarray
    scales: np.ndarray
    bits: int
    group_size: int

    @property
    def estimated_packed_bytes(self) -> int:
        """Ideal packed payload + float32 scales; excludes alignment and metadata."""
        return (self.codes.size * self.bits + 7) // 8 + self.scales.size * 4


def quantize(weights: np.ndarray, *, bits: int = 4, group_size: int = 64) -> QuantizedWeights:
    """Round to nearest (ties-to-even); zero groups use scale 1, no padding codes."""
    positive_int(group_size, "group_size")
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim < 1 or weights.size == 0 or not np.isfinite(weights).all():
        raise ValueError("weights must be a nonempty finite array")
    width = weights.shape[-1]
    scales = np.empty((*weights.shape[:-1], (width + group_size - 1) // group_size), np.float32)
    codes = np.empty(weights.shape, np.int8)
    qmax = 2 ** (bits - 1) - 1
    for group, start in enumerate(range(0, width, group_size)):
        block = weights[..., start : start + group_size]
        maximum = np.max(np.abs(block), axis=-1)
        scale = np.where(maximum == 0, 1.0, maximum / qmax)
        # Avoid underflow to zero for nonzero float32 subnormal weights.
        scale = np.maximum(scale, np.nextafter(np.float32(0), np.float32(1)))
        scales[..., group] = scale
        codes[..., start : start + group_size] = np.clip(
            np.rint(block / scale[..., None]), -qmax, qmax
        ).astype(np.int8)
    return QuantizedWeights(codes, scales, bits, group_size)


def dequantize(weights: QuantizedWeights) -> np.ndarray:
    restored = np.empty(weights.codes.shape, np.float32)
    for group, start in enumerate(range(0, weights.codes.shape[-1], weights.group_size)):
        restored[..., start : start + weights.group_size] = (
            weights.codes[..., start : start + weights.group_size]
            * weights.scales[..., group, None]
        )
    return restored
