"""Deterministic, in-memory character data for the Week 11 lab."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

# This fixture is intentionally short enough to inspect in a terminal and long
# enough to contain recurring character-level patterns. It is not Shakespeare
# and is never downloaded; the optional real-text experiment is documented in
# README.md rather than hidden in the test path.
TINY_TEXT = (
    "recurrent models read one character at a time.\n"
    "a hidden state carries a useful trace of the past.\n"
    "attention lets a decoder look back at the right encoder states.\n"
    "small, deterministic experiments make the gradients easy to inspect.\n"
)


class CharVocab:
    """A stable character-to-index vocabulary with no network dependency."""

    def __init__(self, text: str | Sequence[str]) -> None:
        symbols = sorted(set(text))
        if not symbols:
            raise ValueError("text must contain at least one character")
        self.itos = tuple(symbols)
        self.stoi = {symbol: index for index, symbol in enumerate(self.itos)}

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text).difference(self.stoi))
        if unknown:
            raise ValueError(f"text contains characters absent from vocabulary: {unknown!r}")
        return [self.stoi[symbol] for symbol in text]

    def decode(self, ids: Sequence[int]) -> str:
        try:
            return "".join(self.itos[index] for index in ids)
        except IndexError as error:
            raise ValueError("token id is outside the vocabulary") from error


class TinyCharDataset(Dataset[tuple[Tensor, Tensor]]):
    """Overlapping next-character windows from a supplied in-memory string.

    ``x`` is a window of ``seq_len`` character ids and ``y`` is the same window
    shifted one character to the right. Keeping this contract explicit makes a
    language-model loss a plain token-wise cross entropy.
    """

    def __init__(
        self,
        text: str,
        *,
        seq_len: int = 32,
        vocab: CharVocab | None = None,
        stride: int = 1,
    ) -> None:
        if seq_len <= 0 or stride <= 0:
            raise ValueError("seq_len and stride must be positive")
        if len(text) <= seq_len:
            raise ValueError("text must contain more than seq_len characters")
        self.text = text
        self.seq_len = seq_len
        self.stride = stride
        self.vocab = vocab or CharVocab(text)
        encoded = self.vocab.encode(text)
        self.tokens = torch.tensor(encoded, dtype=torch.long)
        self.starts = tuple(range(0, len(encoded) - seq_len, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = self.starts[index]
        window = self.tokens[start : start + self.seq_len + 1]
        return window[:-1].clone(), window[1:].clone()


def make_tiny_dataset(
    *,
    text: str = TINY_TEXT,
    seq_len: int = 32,
    repeats: int = 1,
    stride: int = 1,
) -> TinyCharDataset:
    """Build a deterministic dataset without reading files or contacting a network."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    expanded = text * repeats
    return TinyCharDataset(expanded, seq_len=seq_len, stride=stride)


# Descriptive aliases make the teaching API convenient without duplicating code.
CharDataset = TinyCharDataset
make_char_dataset = make_tiny_dataset
