"""Deterministic, in-memory character data for the Week 13 lab."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

TINY_TEXT = (
    "small models learn one next character at a time. "
    "causal attention reads the past; a state space model carries a state. "
)


class CharVocab:
    """A deterministic character vocabulary with no special external tokens."""

    def __init__(self, text: str) -> None:
        if not text:
            raise ValueError("text must not be empty")
        self.itos = tuple(sorted(set(text)))
        self.stoi = {symbol: index for index, symbol in enumerate(self.itos)}

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[symbol] for symbol in text]
        except KeyError as error:
            raise ValueError(f"character {error.args[0]!r} is not in this vocabulary") from error

    def decode(self, ids: Sequence[int]) -> str:
        if any(index < 0 or index >= self.size for index in ids):
            raise ValueError("an id is outside the vocabulary")
        return "".join(self.itos[index] for index in ids)


class TinyCharDataset(Dataset[tuple[Tensor, Tensor]]):
    """Overlapping next-character windows held entirely in memory."""

    def __init__(
        self,
        text: str = TINY_TEXT,
        *,
        seq_len: int = 32,
        repeats: int = 2,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        if len(text) < 2:
            raise ValueError("text must contain at least two characters")
        self.text = text * repeats
        if len(self.text) < seq_len + 1:
            raise ValueError("text must contain at least seq_len + 1 characters")
        self.seq_len = seq_len
        self.vocab = CharVocab(self.text)
        encoded = torch.tensor(self.vocab.encode(self.text), dtype=torch.long)
        self._windows = encoded.unfold(0, seq_len + 1, 1)

    def __len__(self) -> int:
        return self._windows.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        window = self._windows[index]
        return window[:-1].clone(), window[1:].clone()


def make_tiny_dataset(*, seq_len: int = 32, repeats: int = 2) -> TinyCharDataset:
    """Return the canonical fixture with stable vocabulary and windows."""

    return TinyCharDataset(seq_len=seq_len, repeats=repeats)
