"""A small decoder-only transformer implemented from elementary PyTorch layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """The fixed sine/cosine position code from the original Transformer."""

    encoding: Tensor

    def __init__(self, d_model: int, max_seq_len: int = 256) -> None:
        super().__init__()
        if d_model <= 0 or max_seq_len <= 0:
            raise ValueError("d_model and max_seq_len must be positive")
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_seq_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        # A final odd feature has no matching cosine frequency.
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (batch, time, features)")
        if embeddings.shape[-1] != self.encoding.shape[-1]:
            raise ValueError("embedding width does not match d_model")
        if embeddings.shape[1] > self.encoding.shape[1]:
            raise ValueError("sequence is longer than max_seq_len")
        return embeddings + self.encoding[:, : embeddings.shape[1]].to(embeddings)


PositionalEncoding = SinusoidalPositionalEncoding


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention whose query at t can only read positions <= t."""

    causal_mask: Tensor

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query_key_value = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )
        self.last_attention: Tensor | None = None

    def forward(
        self, x: Tensor, *, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.d_model or x.shape[1] == 0:
            raise ValueError("x must have shape (batch, nonempty time, d_model)")
        batch, steps, _ = x.shape
        if steps > self.causal_mask.shape[0]:
            raise ValueError("sequence is longer than max_seq_len")
        query, key, value = self.query_key_value(x).chunk(3, dim=-1)
        query = query.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~self.causal_mask[:steps, :steps], torch.finfo(scores.dtype).min
        )
        probabilities = torch.softmax(scores, dim=-1)
        self.last_attention = probabilities
        attended = self.dropout(probabilities) @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, steps, self.d_model)
        result = self.output(attended)
        if return_attention:
            return result, probabilities
        return result

    def attention_probabilities(self) -> Tensor:
        """Return the most recent probabilities for an inspection notebook."""

        if self.last_attention is None:
            raise RuntimeError("run a forward pass before requesting attention")
        return self.last_attention


class MLP(nn.Module):
    """Position-wise feed-forward sublayer."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-normalized attention plus an MLP, with two residual paths."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        *,
        dropout: float = 0.0,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model, n_heads, dropout=dropout, max_seq_len=max_seq_len
        )
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


@dataclass(frozen=True)
class GPTConfig:
    """Small configuration object useful when writing an experiment manifest."""

    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int | None = None
    max_seq_len: int = 64
    dropout: float = 0.0


class NanoGPT(nn.Module):
    """A compact decoder-only language model with no pretrained components."""

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
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or d_model <= 0 or n_layers <= 0:
            raise ValueError("vocab_size, d_model, and n_layers must be positive")
        if d_ff is None:
            d_ff = 4 * d_model
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")
        self.config = GPTConfig(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout)
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        self._validate_ids(input_ids)
        x = self.position(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 20) -> Tensor:
        """Greedily extend ids; this is a qualitative fixture helper, not sampling."""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        # Validate the complete prompt even when a long prompt is later cropped
        # to the model context. Otherwise invalid prefix tokens can be silently
        # accepted, and an empty prompt can reach an invalid last-position read.
        self._validate_ids(input_ids)
        modes = {module: module.training for module in self.modules()}
        self.eval()
        try:
            result = input_ids.clone()
            for _ in range(max_new_tokens):
                context = result[:, -self.config.max_seq_len :]
                next_id = self(context)[:, -1].argmax(dim=-1, keepdim=True)
                result = torch.cat([result, next_id], dim=1)
            return result
        finally:
            for module, training in modes.items():
                module.train(training)

    def _validate_ids(self, input_ids: Tensor) -> None:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one time step")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.vocab_size
        ):
            raise ValueError("input_ids contains an id outside the vocabulary")


# Descriptive aliases make the same implementation easy to find from a lesson.
GPTLanguageModel = NanoGPT
TransformerLanguageModel = NanoGPT
NanoGPTLM = NanoGPT
