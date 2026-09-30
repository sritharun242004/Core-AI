"""A deterministic, dependency-free byte-ish character BPE tokenizer."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path

DEFAULT_SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


class BPETokenizer:
    """Train and use a small byte-free BPE tokenizer entirely in memory.

    The implementation starts with Unicode characters instead of a downloaded
    vocabulary or a regex pre-tokenizer. Merges are selected by frequency and
    then by the lexicographic pair, which makes a run reproducible across
    Python processes. A merged symbol is the concatenation of its children, so
    decoding is lossless for every character observed during training.
    """

    def __init__(
        self,
        vocabulary: Sequence[str],
        merges: Sequence[tuple[str, str]] = (),
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        if not vocabulary:
            raise ValueError("vocabulary must not be empty")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("vocabulary entries must be unique")
        if not special_tokens or len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special_tokens must be non-empty and unique")
        overlap = set(vocabulary).intersection(special_tokens)
        if overlap:
            raise ValueError(f"special tokens already in vocabulary: {sorted(overlap)}")
        self.vocabulary = tuple(vocabulary)
        self.merges = tuple((left, right) for left, right in merges)
        self.special_tokens = tuple(special_tokens)
        self.token_to_id = {
            token: index for index, token in enumerate((*self.special_tokens, *self.vocabulary))
        }
        self.id_to_token = tuple((*self.special_tokens, *self.vocabulary))
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._special_by_length = tuple(sorted(self.special_tokens, key=len, reverse=True))

    @classmethod
    def train(
        cls,
        corpus: Iterable[str],
        *,
        vocab_size: int = 256,
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> BPETokenizer:
        """Learn at most ``vocab_size`` total non-special symbols.

        ``corpus`` is materialised once so generators are accepted without
        changing the deterministic result. Each document starts with its own
        character sequence; no merge can accidentally cross a document edge.
        """

        documents = tuple(corpus)
        if not documents or any(
            not isinstance(document, str) or not document for document in documents
        ):
            raise ValueError("corpus must contain at least one non-empty string")
        if vocab_size < len(special_tokens) + 1:
            raise ValueError("vocab_size must leave room for at least one symbol")
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special_tokens must be unique")

        base_symbols = sorted({character for document in documents for character in document})
        capacity = vocab_size - len(special_tokens)
        if len(base_symbols) > capacity:
            # Keep the most frequent characters first; ties are code-point order.
            counts = Counter(character for document in documents for character in document)
            base_symbols = sorted(base_symbols, key=lambda symbol: (-counts[symbol], symbol))[
                :capacity
            ]
        symbols = list(base_symbols)
        sequences = [
            [character if character in base_symbols else "" for character in document]
            for document in documents
        ]
        # Unknown characters do not occur in the training corpus, so the empty
        # guard is only relevant when vocab_size is smaller than the alphabet.
        sequences = [[symbol for symbol in sequence if symbol] for sequence in sequences]
        merges: list[tuple[str, str]] = []

        while len(symbols) < capacity:
            counts: Counter[tuple[str, str]] = Counter()
            for sequence in sequences:
                counts.update(pairwise(sequence))
            if not counts:
                break
            best_pair = min(counts, key=lambda pair: (-counts[pair], pair[0], pair[1]))
            merged = best_pair[0] + best_pair[1]
            if merged in symbols:
                break
            symbols.append(merged)
            merges.append(best_pair)
            sequences = [cls._merge_sequence(sequence, best_pair, merged) for sequence in sequences]
        return cls(symbols, merges, special_tokens)

    @staticmethod
    def _merge_sequence(sequence: Sequence[str], pair: tuple[str, str], merged: str) -> list[str]:
        result: list[str] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                result.append(merged)
                index += 2
            else:
                result.append(sequence[index])
                index += 1
        return result

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def _split_specials(self, text: str) -> list[str]:
        pieces: list[str] = []
        buffer: list[str] = []
        index = 0
        while index < len(text):
            matched = next(
                (token for token in self._special_by_length if text.startswith(token, index)),
                None,
            )
            if matched is not None:
                pieces.extend(buffer)
                buffer = []
                pieces.append(matched)
                index += len(matched)
            else:
                buffer.append(text[index])
                index += 1
        pieces.extend(buffer)
        return pieces

    def _encode_plain(self, text: str) -> list[str]:
        symbols = [character if character in self.token_to_id else "<unk>" for character in text]
        for pair in self.merges:
            symbols = self._merge_sequence(symbols, pair, pair[0] + pair[1])
        return symbols

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens: list[str] = []
        for piece in self._split_specials(text):
            if piece in self.special_tokens:
                tokens.append(piece)
            else:
                tokens.extend(self._encode_plain(piece))
        if add_bos:
            tokens.insert(0, "<bos>")
        if add_eos:
            tokens.append("<eos>")
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        decoded: list[str] = []
        for index in ids:
            if not isinstance(index, int) or index < 0 or index >= self.vocab_size:
                raise ValueError(f"token id {index!r} is outside the vocabulary")
            token = self.id_to_token[index]
            if token in self.special_tokens:
                if not skip_special_tokens:
                    decoded.append(token)
            else:
                decoded.append(token)
        return "".join(decoded)

    def to_dict(self) -> dict[str, object]:
        return {
            "special_tokens": list(self.special_tokens),
            "vocabulary": list(self.vocabulary),
            "merges": [list(pair) for pair in self.merges],
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BPETokenizer:
        vocabulary = payload.get("vocabulary")
        merges = payload.get("merges", [])
        special_tokens = payload.get("special_tokens", DEFAULT_SPECIAL_TOKENS)
        if not isinstance(vocabulary, list) or not all(
            isinstance(item, str) for item in vocabulary
        ):
            raise ValueError("tokenizer vocabulary must be a list of strings")
        if not isinstance(special_tokens, list) or not all(
            isinstance(item, str) for item in special_tokens
        ):
            raise ValueError("tokenizer special_tokens must be a list of strings")
        if not isinstance(merges, list):
            raise ValueError("tokenizer merges must be a list")
        parsed_merges = []
        for pair in merges:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(item, str) for item in pair)
            ):
                raise ValueError("each tokenizer merge must contain two strings")
            parsed_merges.append((pair[0], pair[1]))
        return cls(vocabulary, parsed_merges, special_tokens)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict):
            raise ValueError("tokenizer JSON must contain an object")
        return cls.from_dict(payload)


MiniBPETokenizer = BPETokenizer
