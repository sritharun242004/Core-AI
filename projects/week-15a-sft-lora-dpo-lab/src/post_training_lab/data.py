"""Small deterministic preference and supervised fixtures kept entirely in memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SFTExample:
    """One already-tokenized next-token training example.

    Keeping tokenization outside the model makes the objective and its masks
    visible. ``labels`` uses ``-100`` for positions that should be ignored.
    """

    input_ids: Tensor
    labels: Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 1 or self.labels.ndim != 1:
            raise ValueError("SFTExample tensors must be one-dimensional")
        if self.input_ids.shape != self.labels.shape:
            raise ValueError("input_ids and labels must have the same shape")
        if self.input_ids.dtype != torch.long or self.labels.dtype != torch.long:
            raise TypeError("SFTExample tensors must have dtype torch.long")


@dataclass(frozen=True)
class PreferencePair:
    """Token ids for a prompt and its chosen/rejected full responses."""

    prompt: str
    chosen: tuple[int, ...]
    rejected: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.prompt or not self.chosen or not self.rejected:
            raise ValueError("preference fields must not be empty")
        if any(token < 0 for token in (*self.chosen, *self.rejected)):
            raise ValueError("preference token ids must be non-negative")

    @property
    def chosen_ids(self) -> Tensor:
        return torch.tensor(self.chosen, dtype=torch.long)

    @property
    def rejected_ids(self) -> Tensor:
        return torch.tensor(self.rejected, dtype=torch.long)


def make_sft_examples(
    *, vocab_size: int = 16, count: int = 4, seq_len: int = 8
) -> list[SFTExample]:
    """Return a stable tiny instruction-style next-token fixture.

    The repeated cycle gives a real gradient signal while remaining small
    enough for a CPU unit test. ``vocab_size`` is accepted to make the fixture
    convenient for differently sized toy vocabularies.
    """

    if vocab_size < 10 or count < 1 or seq_len < 2:
        raise ValueError("vocab_size must be >= 10, count positive, and seq_len >= 2")
    base = torch.arange(1, seq_len + 1, dtype=torch.long) % vocab_size
    examples: list[SFTExample] = []
    for offset in range(count):
        input_ids = ((base + offset) % vocab_size).clone()
        labels = ((input_ids + 1) % vocab_size).clone()
        examples.append(SFTExample(input_ids=input_ids, labels=labels))
    return examples


def make_preference_pairs() -> list[PreferencePair]:
    """Return deterministic pairs where the first response is preferred.

    The ids are intentionally not tied to a tokenizer. They are illustrative
    categorical tokens for objective plumbing and can be fed to ``TinyCausalLM``
    after adding a prompt prefix in a notebook or assignment.
    """

    return [
        PreferencePair("answer arithmetic", (1, 4, 8, 2), (1, 4, 9, 2)),
        PreferencePair("answer concise", (1, 5, 7, 2), (1, 5, 10, 2)),
        PreferencePair("answer safe", (1, 6, 11, 2), (1, 6, 12, 2)),
        PreferencePair("answer grounded", (1, 3, 13, 2), (1, 3, 14, 2)),
    ]
