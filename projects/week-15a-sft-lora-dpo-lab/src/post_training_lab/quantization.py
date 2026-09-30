"""Optional educational int8 packing helpers (not a production quantizer)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class QuantizedTensor:
    """Symmetric per-tensor int8 values and the scale needed to restore them."""

    values: Tensor
    scale: Tensor

    def __iter__(self):
        # Permit the familiar ``values, scale = quantize_int8(weight)`` form.
        yield self.values
        yield self.scale


def quantize_int8(weight: Tensor) -> QuantizedTensor:
    """Pack a floating tensor with symmetric per-tensor int8 quantization."""

    if not weight.is_floating_point() or weight.numel() == 0:
        raise ValueError("weight must be a non-empty floating tensor")
    scale = weight.detach().abs().amax() / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    values = torch.clamp(torch.round(weight.detach() / scale), -127, 127).to(torch.int8)
    return QuantizedTensor(values=values, scale=scale.to(dtype=weight.dtype))


def dequantize_int8(
    packed: QuantizedTensor | Tensor, scale: Tensor | float | None = None
) -> Tensor:
    """Restore an int8 tensor; accepts either the dataclass or values plus scale."""

    if isinstance(packed, QuantizedTensor):
        values, packed_scale = packed.values, packed.scale
        if scale is None:
            scale = packed_scale
    else:
        values = packed
    if scale is None:
        raise ValueError("scale is required when passing raw int8 values")
    return values.to(torch.get_default_dtype()) * torch.as_tensor(
        scale, dtype=torch.get_default_dtype()
    )


def qlora_recipe() -> dict[str, str]:
    """Explain the local QLoRA boundary without requiring bitsandbytes or CUDA."""

    return {
        "base_weights": "int8 teaching representation",
        "trainable": "LoRA A/B matrices only",
        "compute": "dequantize for the forward pass; no CUDA kernel required",
        "boundary": "not NF4, paged optimizers, or a benchmark of production QLoRA",
    }
