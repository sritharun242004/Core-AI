"""Tiny deterministic causal-language-model pretraining helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from ._torch import backward, manual_seed
from .data import TinyStoriesDataset
from .model import TinyCausalLM
from .tokenizer import BPETokenizer


@dataclass
class TrainResult:
    losses: list[float]
    steps: int
    tokens_seen: int


def causal_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("logits must be (batch,time,vocab), targets must be (batch,time)")
    if logits.shape[:2] != targets.shape:
        raise ValueError("logits and targets must share batch and time dimensions")
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )


def pretrain(
    model: TinyCausalLM,
    dataset: TinyStoriesDataset,
    *,
    steps: int = 100,
    batch_size: int = 8,
    learning_rate: float = 3e-3,
    optimizer: torch.optim.Optimizer | None = None,
    seed: int = 0,
) -> TrainResult:
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")
    if model.vocab_size != dataset.tokenizer.vocab_size:
        raise ValueError("model vocabulary must match dataset tokenizer")
    manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    losses: list[float] = []
    for _ in range(steps):
        inputs, targets = dataset.sample_batch(batch_size, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = causal_cross_entropy(model(inputs), targets)
        backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        cast(Callable[[], None], optimizer.step)()
        losses.append(float(loss.detach()))
    return TrainResult(losses=losses, steps=steps, tokens_seen=steps * batch_size * dataset.seq_len)


def save_checkpoint(
    path: str | Path,
    model: TinyCausalLM,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    tokenizer: BPETokenizer | None = None,
    history: list[float] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "config": model.config.as_dict(),
        "model": model.state_dict(),
        "step": step,
        "history": list(history or []),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if tokenizer is not None:
        payload["tokenizer"] = tokenizer.to_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: TinyCausalLM,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    loaded: object = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict) or "model" not in loaded:
        raise ValueError("checkpoint must contain a model state")
    payload = cast(dict[str, Any], loaded)
    raw_state: object = payload["model"]
    if not isinstance(raw_state, dict):
        raise ValueError("checkpoint model state must be a mapping")
    state: dict[str, Tensor] = {}
    for key, value in cast(dict[object, object], raw_state).items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise ValueError("checkpoint model state must map names to tensors")
        state[key] = value
    model.load_state_dict(state)
    if optimizer is not None and "optimizer" in payload:
        raw_optimizer: object = payload["optimizer"]
        if not isinstance(raw_optimizer, dict):
            raise ValueError("checkpoint optimizer state must be a mapping")
        optimizer.load_state_dict(cast(dict[str, Any], raw_optimizer))
    return payload


__all__ = ["TrainResult", "causal_cross_entropy", "load_checkpoint", "pretrain", "save_checkpoint"]
