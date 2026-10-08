# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownLambdaType=false, reportCallIssue=false, reportUnnecessaryIsInstance=false, reportIndexIssue=false, reportPrivateUsage=false, reportMissingTypeArgument=false, reportUnknownParameterType=false, reportMissingImports=false, reportPossiblyUnboundVariable=false
"""Minimal LoRA injection, with no dependency on PEFT or bitsandbytes."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """A linear map plus ``B(A(x)) * alpha/r``; base freezing is configurable."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        *,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        _validate_lora_options(rank, alpha, dropout)
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.train(base.training)
        if freeze_base:
            _freeze_parameters(self.base.parameters())

    def forward(self, x: Tensor) -> Tensor:
        update = self.dropout(x) @ self.lora_A.transpose(0, 1)
        update = update @ self.lora_B.transpose(0, 1)
        return self.base(x) + self.scaling * update

    def merged_weight(self) -> Tensor:
        """Return base + update for export or inference inspection."""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


def _replace_child(parent: nn.Module, name: str, value: nn.Module) -> None:
    if isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(name)] = value
    else:
        setattr(parent, name, value)


def _validate_lora_options(rank: int, alpha: float, dropout: float) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    if not math.isfinite(float(alpha)) or float(alpha) < 0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(float(dropout)) or not 0 <= float(dropout) <= 1:
        raise ValueError("dropout must be finite and between zero and one")


def _freeze_parameters(parameters: Iterable[nn.Parameter]) -> None:
    for parameter in parameters:
        parameter.requires_grad = False
        parameter.grad = None


def _freeze_model_base(model: nn.Module) -> None:
    """Freeze dense parameters while preserving existing adapter parameters."""

    adapters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in (module.lora_A, module.lora_B)
    }
    for parameter in model.parameters():
        if id(parameter) not in adapters:
            _freeze_parameters((parameter,))


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float | None = None,
    target_modules: Iterable[str] = ("q_proj", "v_proj"),
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> nn.Module:
    """Replace named ``nn.Linear`` children with ``LoRALinear`` modules.

    ``target_modules`` matches a leaf module name, so the default only adapts
    query/value projections. The function mutates and returns ``model`` to make
    notebook usage concise. A zero-initialized B matrix preserves the original
    model's logits before the first adapter update.
    """

    names = set(target_modules)
    if not names:
        raise ValueError("target_modules must not be empty")
    if alpha is None:
        alpha = float(rank)
    _validate_lora_options(rank, alpha, dropout)

    # Discover all targets before changing requires_grad or module ownership.
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for _parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in names and isinstance(child, nn.Linear):
                replacements.append((parent, child_name, child))
    if not replacements:
        raise ValueError(f"no Linear modules matched target_modules={sorted(names)}")
    if freeze_base:
        _freeze_model_base(model)
    for parent, child_name, child in replacements:
        _replace_child(
            parent,
            child_name,
            LoRALinear(child, rank, alpha, dropout, freeze_base=freeze_base),
        )
    return model


def mark_only_lora_trainable(model: nn.Module) -> int:
    """Freeze all parameters except actual LoRA A/B parameters; return count."""

    adapter_parameters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in (module.lora_A, module.lora_B)
    }
    trainable = 0
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in adapter_parameters
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            parameter.grad = None
    return trainable


def lora_config(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Return portable adapter-shape/config metadata for checkpoint validation."""

    return {
        name: {
            "rank": module.rank,
            "alpha": module.alpha,
            "dropout": module.dropout.p,
            "in_features": module.base.in_features,
            "out_features": module.base.out_features,
        }
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
