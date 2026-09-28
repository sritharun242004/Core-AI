"""Character LSTM encoder-decoder with Bahdanau-style additive attention."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .data import CharVocab


class AdditiveAttention(nn.Module):
    """Learned additive attention over ``values``.

    For every decoder query ``q_t`` and encoder state ``h_s`` this computes
    ``e[t,s] = v(tanh(W_q q_t + W_h h_s))`` and normalizes over source time.
    Queries may be ``(B, H)`` or ``(B, T, H)``; values are ``(B, S, H)``.
    """

    def __init__(self, hidden_size: int, attention_size: int | None = None) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if attention_size is None:
            attention_size = hidden_size
        if attention_size <= 0:
            raise ValueError("attention_size must be positive")
        self.query_projection = nn.Linear(hidden_size, attention_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, attention_size, bias=False)
        self.score_projection = nn.Linear(attention_size, 1, bias=False)

    def forward(
        self,
        query: Tensor,
        values: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if values.ndim != 3:
            raise ValueError("values must have shape (batch, source_steps, hidden_size)")
        query_was_unbatched_time = query.ndim == 2
        if query_was_unbatched_time:
            query = query.unsqueeze(1)
        if query.ndim != 3:
            raise ValueError(
                "query must have shape (batch, hidden_size) or (batch, steps, hidden_size)"
            )
        if query.shape[0] != values.shape[0] or query.shape[-1] != values.shape[-1]:
            raise ValueError("query and values must agree on batch and hidden dimensions")

        query_term = self.query_projection(query).unsqueeze(2)
        value_term = self.value_projection(values).unsqueeze(1)
        energies = self.score_projection(torch.tanh(query_term + value_term)).squeeze(-1)
        if mask is not None:
            if mask.shape != values.shape[:2]:
                raise ValueError("mask must have shape (batch, source_steps)")
            energies = energies.masked_fill(
                ~mask[:, None, :].bool(), torch.finfo(energies.dtype).min
            )
        weights = torch.softmax(energies, dim=-1)
        context = torch.bmm(weights, values)
        if query_was_unbatched_time:
            return context[:, 0], weights[:, 0]
        return context, weights


class CharLSTMAttention(nn.Module):
    """A compact character seq2seq model with an additive attention bridge.

    The encoder reads a source character window. The decoder reads either the
    source window (language-model convenience) or teacher-forced target tokens,
    attends over every encoder state, and predicts one vocabulary distribution
    per decoder step. Thus a training batch has logits ``(B, T, V)`` and
    attention weights ``(B, T, S)``. ``rnn_type`` can be ``"lstm"`` (the
    default) or ``"gru"``; both recurrent cells use the same attention and
    shape contract.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_size: int = 64,
        hidden_size: int = 128,
        attention_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        rnn_type: Literal["lstm", "gru"] = "lstm",
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or embedding_size <= 0 or hidden_size <= 0:
            raise ValueError("vocab_size, embedding_size, and hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        recurrent = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.encoder = recurrent(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.decoder = recurrent(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.attention = AdditiveAttention(hidden_size, attention_size)
        self.output = nn.Linear(hidden_size * 2, vocab_size)

    def forward(
        self,
        source_ids: Tensor,
        target_ids: Tensor | None = None,
        *,
        teacher_forcing_ratio: float = 1.0,
        source_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._validate_ids(source_ids, "source_ids")
        if target_ids is not None:
            self._validate_ids(target_ids, "target_ids")
            if target_ids.shape[0] != source_ids.shape[0]:
                raise ValueError("source_ids and target_ids must have the same batch size")
        if not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")

        encoder_outputs, state = self.encoder(self.embedding(source_ids))
        if target_ids is None or teacher_forcing_ratio >= 1:
            decoder_inputs = source_ids
            if target_ids is not None:
                # The first source token acts as a tiny BOS token; later
                # decoder inputs are the previous gold target character.
                decoder_inputs = torch.cat([source_ids[:, :1], target_ids[:, :-1]], dim=1)
            decoder_outputs, _ = self.decoder(self.embedding(decoder_inputs), state)
            contexts, weights = self.attention(decoder_outputs, encoder_outputs, source_mask)
            logits = self.output(torch.cat([decoder_outputs, contexts], dim=-1))
            return logits, weights

        # A partial ratio is genuinely mixed teacher forcing rather than an
        # ignored argument: after each prediction, choose the gold previous
        # token or the model's detached argmax for the next decoder input.
        decoder_input = source_ids[:, :1]
        decoder_state = state
        logits_steps: list[Tensor] = []
        weight_steps: list[Tensor] = []
        for step in range(target_ids.shape[1]):
            decoder_output, decoder_state = self.decoder(
                self.embedding(decoder_input), decoder_state
            )
            context, step_weights = self.attention(
                decoder_output, encoder_outputs, source_mask
            )
            step_logits = self.output(torch.cat([decoder_output, context], dim=-1))
            logits_steps.append(step_logits)
            weight_steps.append(step_weights)
            if step + 1 < target_ids.shape[1]:
                predicted = step_logits[:, -1].argmax(dim=-1)
                if teacher_forcing_ratio == 0:
                    decoder_input = predicted.detach().unsqueeze(1)
                else:
                    use_teacher = torch.rand(
                        source_ids.shape[0], device=source_ids.device
                    ) < teacher_forcing_ratio
                    next_input = torch.where(
                        use_teacher, target_ids[:, step], predicted.detach()
                    )
                    decoder_input = next_input.unsqueeze(1)
        return torch.cat(logits_steps, dim=1), torch.cat(weight_steps, dim=1)

    def generate(
        self,
        vocab: CharVocab,
        prompt: str,
        *,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        device: str | torch.device | None = None,
    ) -> str:
        """Greedily or stochastically continue a prompt without any I/O."""

        if not prompt:
            raise ValueError("prompt must contain at least one character")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        selected = next(self.parameters()).device if device is None else torch.device(device)
        ids = torch.tensor([vocab.encode(prompt)], dtype=torch.long, device=selected)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self(ids)
                next_logits = logits[:, -1, :]
                if temperature == 0:
                    next_id = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    probabilities = torch.softmax(next_logits / temperature, dim=-1)
                    next_id = torch.multinomial(probabilities, num_samples=1)
                ids = torch.cat([ids, next_id], dim=1)
        if was_training:
            self.train()
        return vocab.decode(ids[0].tolist())

    def _validate_ids(self, ids: Tensor, name: str) -> None:
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError(f"{name} must be a rank-2 torch.long tensor")
        if ids.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one time step")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.vocab_size):
            raise ValueError(f"{name} contains an id outside [0, {self.vocab_size})")


class CharGRUAttention(CharLSTMAttention):
    """GRU encoder-decoder variant sharing the additive-attention contract."""

    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_size: int = 64,
        hidden_size: int = 128,
        attention_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            attention_size=attention_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type="gru",
        )


# Short aliases keep imports natural in a lesson that calls the architecture a
# character RNN while making the concrete recurrent cells explicit.
CharRNN = CharLSTMAttention
CharLSTM = CharLSTMAttention
CharGRU = CharGRUAttention
BahdanauAttention = AdditiveAttention
additive_attention = AdditiveAttention
