# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownLambdaType=false, reportCallIssue=false
"""Tiny supervised trainer and portable checkpoints for the offline lab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ._torch import backward, manual_seed
from .data import SFTExample
from .lora import lora_config
from .objectives import sft_loss


def _batch(examples: list[SFTExample], device: torch.device) -> tuple[Tensor, Tensor]:
    if not examples:
        raise ValueError("examples must not be empty")
    return (
        torch.stack([example.input_ids for example in examples]).to(device),
        torch.stack([example.labels for example in examples]).to(device),
    )


def train_sft(
    model: nn.Module,
    examples: list[SFTExample],
    *,
    epochs: int = 5,
    learning_rate: float = 0.05,
    batch_size: int | None = None,
    seed: int | None = None,
) -> dict[str, list[float]]:
    """Run deterministic full-batch (or mini-batch) SFT and return loss history."""

    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs must be positive and learning_rate must be positive")
    if seed is not None:
        manual_seed(seed)
    if batch_size is None:
        batch_size = len(examples)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    device = next(model.parameters()).device
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        epoch_losses: list[Tensor] = []
        for start in range(0, len(examples), batch_size):
            inputs, labels = _batch(examples[start : start + batch_size], device)
            optimizer.zero_grad(set_to_none=True)
            loss = sft_loss(model(inputs), labels)
            backward(loss)
            optimizer.step()
            epoch_losses.append(loss.detach())
        history.append(float(torch.stack(epoch_losses).mean()))
    return {"loss": history}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | str | Path | None = None,
    path: str | Path = "checkpoint.pt",
    *,
    step: int = 0,
    **metadata: Any,
) -> None:
    """Save model, optional optimizer, step, and JSON-like metadata on CPU."""

    if isinstance(optimizer, (str, Path)):
        path, optimizer = optimizer, None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if "_requires_grad" in metadata or "_lora_config" in metadata:
        raise ValueError("reserved checkpoint metadata keys cannot be overridden")
    payload: dict[str, Any] = {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "step": int(step),
        "_requires_grad": [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ],
        "_lora_config": lora_config(model),
        **metadata,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, target)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | str | Path | None = None,
    path: str | Path = "checkpoint.pt",
) -> dict[str, Any]:
    """Restore a checkpoint and return its non-state metadata."""

    if isinstance(optimizer, (str, Path)):
        path, optimizer = optimizer, None
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing a model state dictionary")
    current = model.state_dict()
    if set(state) != set(current) or any(
        state[name].shape != current[name].shape for name in state if name in current
    ):
        raise ValueError("checkpoint model state is incompatible, including its LoRA architecture")
    saved_lora = payload.get("_lora_config")
    if saved_lora is not None and saved_lora != lora_config(model):
        raise ValueError("checkpoint LoRA configuration does not match the target model")
    model.load_state_dict(state)
    saved_trainable = payload.get("_requires_grad")
    if saved_trainable is not None:
        if not isinstance(saved_trainable, list):
            raise ValueError("checkpoint trainability metadata must be a list")
        trainable_names = set(saved_trainable)
        current_names = set(dict(model.named_parameters()))
        if not trainable_names <= current_names:
            raise ValueError("checkpoint trainability metadata does not match the target model")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name in trainable_names
            if not parameter.requires_grad:
                parameter.grad = None
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return {
        key: value
        for key, value in payload.items()
        if key not in {"model", "optimizer", "_requires_grad", "_lora_config"}
    }
