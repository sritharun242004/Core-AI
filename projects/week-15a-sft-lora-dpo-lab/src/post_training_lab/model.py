# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownLambdaType=false, reportCallIssue=false, reportIndexIssue=false
"""A deliberately small causal language model for CPU/offline experiments."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with named projections for LoRA injection."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:length, :length], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.out_proj(attended)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyCausalLM(nn.Module):
    """Tiny decoder-only LM; logits at position ``t`` predict token ``t+1``.

    It is intentionally small enough to run in a unit test. There are no
    tokenizers, downloads, or framework-specific model wrappers in this lab.
    """

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 1,
        max_seq_len: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, max_seq_len, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying is useful in a toy model and keeps the checkpoint compact.
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, length = input_ids.shape
        if length > self.max_seq_len:
            raise ValueError(f"sequence length {length} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Hugging Face-style alias used in the lesson and exercises."""
        return self.parameter_count(trainable_only=trainable_only)


# The same decoder is a policy once its log-probabilities are compared by a
# preference objective. Keep the alias readable for notebook discussions.
TinyPolicy = TinyCausalLM
