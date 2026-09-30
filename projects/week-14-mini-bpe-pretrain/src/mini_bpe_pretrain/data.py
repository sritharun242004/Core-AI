"""Small in-memory TinyStories-shaped fixtures and next-token windows."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .tokenizer import BPETokenizer

TINY_STORIES_CORPUS = (
    "the little fox found a blue kite.",
    "the fox ran home and showed the kite to mia.",
    "mia smiled because the kite could fly high.",
    "a small dog watched the fox and wagged its tail.",
    "the friends shared a warm cookie under the tree.",
    "when the sun went down, they walked home together.",
)


def make_tinystories_corpus() -> tuple[str, ...]:
    """Return a fresh tuple so callers cannot mutate the canonical fixture."""

    return tuple(TINY_STORIES_CORPUS)


class TinyStoriesDataset(Dataset[tuple[Tensor, Tensor]]):
    """Overlapping token windows over a tiny story-shaped corpus.

    The data intentionally remains a handful of short, synthetic stories. It
    exercises document boundaries and natural-language punctuation without
    implying that a toy fixture is the TinyStories dataset or a quality
    benchmark.
    """

    def __init__(
        self,
        corpus: Sequence[str] | None = None,
        *,
        seq_len: int = 32,
        vocab_size: int = 96,
        tokenizer: BPETokenizer | None = None,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        documents = tuple(make_tinystories_corpus() if corpus is None else corpus)
        if not documents or any(not document for document in documents):
            raise ValueError("corpus must contain non-empty documents")
        self.corpus = documents
        self.seq_len = seq_len
        self.tokenizer = tokenizer or BPETokenizer.train(documents, vocab_size=vocab_size)
        encoded: list[int] = []
        for document in documents:
            encoded.extend(self.tokenizer.encode(document, add_bos=True, add_eos=True))
        if len(encoded) < seq_len + 1:
            raise ValueError("corpus must contain at least seq_len + 1 tokens")
        stream = torch.tensor(encoded, dtype=torch.long)
        self._windows = stream.unfold(0, seq_len + 1, 1)

    def __len__(self) -> int:
        return int(self._windows.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        window = self._windows[index]
        return window[:-1].clone(), window[1:].clone()

    def batch(self, indices: Sequence[int]) -> tuple[Tensor, Tensor]:
        """Stack selected windows without a DataLoader or worker randomness."""

        if not indices:
            raise ValueError("indices must not be empty")
        inputs, targets = zip(*(self[index] for index in indices), strict=True)
        return torch.stack(inputs), torch.stack(targets)

    def sample_batch(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = torch.randint(len(self), (batch_size,), generator=generator).tolist()
        return self.batch(indices)


make_tiny_stories_dataset = TinyStoriesDataset
