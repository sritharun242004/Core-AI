"""A tiny decoder-only language model with RoPE or ALiBi positions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .position import apply_rope, build_alibi_bias, build_rope_cache


@dataclass(frozen=True)
class LMConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int | None = None
    max_seq_len: int = 64
    dropout: float = 0.0
    position_type: str = "rope"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with an inspectable probability tensor."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        *,
        position_type: str = "rope",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or max_seq_len <= 0:
            raise ValueError("d_model, n_heads, and max_seq_len must be positive")
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("d_model must divide by n_heads and give an even head dimension")
        if position_type not in {"rope", "alibi"}:
            raise ValueError("position_type must be 'rope' or 'alibi'")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.position_type = position_type
        self.query_key_value = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )
        if position_type == "rope":
            cos, sin = build_rope_cache(max_seq_len, self.head_dim)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.register_buffer("alibi", build_alibi_bias(n_heads, max_seq_len), persistent=False)
        self.last_attention: Tensor | None = None

    def forward(
        self, x: Tensor, *, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.d_model or x.shape[1] == 0:
            raise ValueError("x must have shape (batch, nonempty time, d_model)")
        batch, steps, _ = x.shape
        if steps > self.max_seq_len:
            raise ValueError("sequence is longer than max_seq_len")
        query, key, value = self.query_key_value(x).chunk(3, dim=-1)
        query = query.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        if self.position_type == "rope":
            query, key = apply_rope(query, key, self.rope_cos, self.rope_sin)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.position_type == "alibi":
            scores = scores + self.alibi[:, :, :steps, :steps].to(scores)
        scores = scores.masked_fill(
            ~self.causal_mask[:steps, :steps], torch.finfo(scores.dtype).min
        )
        probabilities = torch.softmax(scores, dim=-1)
        self.last_attention = probabilities
        attended = self.dropout(probabilities) @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, steps, self.d_model)
        output = self.output(attended)
        return (output, probabilities) if return_attention else output

    def attention_probabilities(self) -> Tensor:
        if self.last_attention is None:
            raise RuntimeError("run a forward pass before requesting attention")
        return self.last_attention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        *,
        position_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model,
            n_heads,
            max_seq_len,
            position_type=position_type,
            dropout=dropout,
        )
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


class TinyCausalLM(nn.Module):
    """Small causal LM suitable for a deterministic CPU pretraining smoke test."""

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int | None = None,
        max_seq_len: int = 64,
        dropout: float = 0.0,
        position_type: str = "rope",
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or d_model <= 0 or n_layers <= 0:
            raise ValueError("vocab_size, d_model, and n_layers must be positive")
        if d_ff is None:
            d_ff = 4 * d_model
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")
        self.config = LMConfig(
            vocab_size,
            d_model,
            n_heads,
            n_layers,
            d_ff,
            max_seq_len,
            dropout,
            position_type,
        )
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    max_seq_len,
                    position_type=position_type,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    @classmethod
    def from_config(cls, config: LMConfig | dict[str, object]) -> TinyCausalLM:
        values = config.as_dict() if isinstance(config, LMConfig) else dict(config)
        return cls(**values)

    def forward(self, input_ids: Tensor) -> Tensor:
        self._validate_ids(input_ids)
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 20) -> Tensor:
        self._validate_ids(input_ids)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        was_training = self.training
        self.eval()
        result = input_ids.clone()
        for _ in range(max_new_tokens):
            context = result[:, -self.max_seq_len :]
            result = torch.cat([result, self(context)[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
        if was_training:
            self.train()
        return result

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    num_parameters = parameter_count

    def _validate_ids(self, input_ids: Tensor) -> None:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        if input_ids.shape[1] == 0 or input_ids.shape[1] > self.max_seq_len:
            raise ValueError("input_ids has an invalid sequence length")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.vocab_size
        ):
            raise ValueError("input_ids contains an id outside the vocabulary")


MiniCausalLM = TinyCausalLM
