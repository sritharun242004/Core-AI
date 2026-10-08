"""A transparent sparse mixture-of-experts implementation for CPU teaching."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class Routing:
    """The inspectable decisions made by a :class:`TopKRouter`.

    All token axes are flattened to ``N = batch * time``. For k > 1,
    ``topk_weights`` sum to one before capacity is applied. For k = 1, the
    selected all-expert probability is retained so task gradients can train
    the router. ``dispatch_mask`` identifies the assignments that fit in each
    expert's capacity; dropped assignments contribute no weighted value.
    """

    probabilities: Tensor
    topk_indices: Tensor
    topk_weights: Tensor
    dispatch_mask: Tensor
    expert_loads: Tensor
    selected_loads: Tensor
    capacity: int
    aux_loss: Tensor


class TopKRouter(nn.Module):
    """Learned top-k router with explicit, deterministic capacity handling."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model < 1 or num_experts < 1:
            raise ValueError("d_model and num_experts must be positive")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between one and num_experts")
        if not math.isfinite(capacity_factor) or capacity_factor <= 0:
            raise ValueError("capacity_factor must be finite and positive")
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = float(capacity_factor)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.last_routing: Routing | None = None

    def forward(self, x: Tensor) -> Routing:
        if x.ndim not in (2, 3):
            raise ValueError(
                "router input must have shape [tokens, d_model] or [batch, time, d_model]"
            )
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] == 0:
            raise ValueError("router needs at least one token")
        logits = self.gate(flat)
        # Keep softmax and balance reductions in at least float32; half/bfloat16
        # counts would round or overflow even at modest teaching batch sizes.
        routing_logits = (
            logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
        )
        probabilities = routing_logits.softmax(dim=-1)
        selected_logits, topk_indices = torch.topk(routing_logits, self.top_k, dim=-1)
        if self.top_k == 1:
            # A one-element softmax is identically one and has zero derivative.
            topk_weights = probabilities.gather(-1, topk_indices)
        else:
            topk_weights = selected_logits.softmax(dim=-1)

        # This is a per-expert token-slot budget. Keeping the first selected
        # tokens makes capacity behavior deterministic and easy to inspect.
        capacity = max(
            1,
            math.ceil(self.capacity_factor * flat.shape[0] * self.top_k / self.num_experts),
        )
        dispatch_mask = torch.zeros_like(topk_indices, dtype=torch.bool)
        for expert_index in range(self.num_experts):
            token_slots = torch.where(topk_indices == expert_index)
            keep = token_slots[0][:capacity]
            slot = token_slots[1][:capacity]
            dispatch_mask[keep, slot] = True
        expert_loads = torch.bincount(topk_indices[dispatch_mask], minlength=self.num_experts)
        selected_loads = torch.bincount(topk_indices.reshape(-1), minlength=self.num_experts)
        # Switch-style importance/load auxiliary loss uses the router's
        # selected assignments before capacity drops. Otherwise a saturated
        # expert could look balanced merely because its excess tokens vanished.
        fractions = selected_loads.to(probabilities.dtype) / (flat.shape[0] * self.top_k)
        importance = probabilities.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(fractions * importance)
        routing = Routing(
            probabilities=probabilities,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            dispatch_mask=dispatch_mask,
            expert_loads=expert_loads,
            selected_loads=selected_loads,
            capacity=capacity,
            aux_loss=aux_loss,
        )
        self.last_routing = routing
        return routing


class ExpertMLP(nn.Module):
    """One position-wise feed-forward expert."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TinyMoE(nn.Module):
    """Sparse top-k MoE whose output preserves ``[batch, time, d_model]``."""

    def __init__(
        self,
        d_model: int = 16,
        hidden_dim: int = 32,
        num_experts: int = 4,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.router = TopKRouter(d_model, num_experts, top_k, capacity_factor)
        self.experts = nn.ModuleList([ExpertMLP(d_model, hidden_dim) for _ in range(num_experts)])
        self.last_routing: Routing | None = None

    def route(self, x: Tensor) -> Routing:
        return self.router(x)

    def forward(self, x: Tensor, return_aux: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("MoE input must have shape [batch, time, d_model]")
        batch, time, _ = x.shape
        flat = x.reshape(-1, self.d_model)
        routing = self.router(flat)
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            token_indices, topk_slots = torch.where(
                routing.dispatch_mask & (routing.topk_indices == expert_index)
            )
            if token_indices.numel() == 0:
                continue
            expert_output = expert(flat[token_indices])
            weights = (
                routing.topk_weights[token_indices, topk_slots].to(expert_output).unsqueeze(-1)
            )
            output.index_add_(0, token_indices, expert_output * weights)
        output = output.reshape(batch, time, self.d_model)
        self.last_routing = routing
        if return_aux:
            return output, routing.aux_loss
        return output

    def aux_loss(self) -> Tensor:
        """Return the latest load-balancing loss, or route a clear error."""

        if self.last_routing is None:
            raise RuntimeError("call the MoE before requesting aux_loss")
        return self.last_routing.aux_loss


class TinyMoEClassifier(nn.Module):
    """Small classifier used to demonstrate supervised MoE optimization."""

    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 16,
        hidden_dim: int = 32,
        num_experts: int = 3,
        num_classes: int = 2,
        top_k: int = 2,
        capacity_factor: float = 1.5,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.moe = TinyMoE(d_model, hidden_dim, num_experts, top_k, capacity_factor)
        self.output = nn.Linear(d_model, num_classes)
        self._aux_loss: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        elif x.ndim == 3:
            squeeze = False
        else:
            raise ValueError(
                "classifier input must have shape [batch, features] or [batch, time, features]"
            )
        hidden = self.input_projection(x)
        hidden, self._aux_loss = self.moe(hidden, return_aux=True)
        logits = self.output(hidden)
        return logits[:, 0] if squeeze else logits

    @property
    def aux_loss(self) -> Tensor:
        if self._aux_loss is None:
            raise RuntimeError("call the classifier before requesting aux_loss")
        return self._aux_loss
