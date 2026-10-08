# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownLambdaType=false
"""Transparent TF-IDF, dense-fixture, hybrid, and reranking baselines."""

import itertools
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .corpus import ALIASES, DOCUMENTS, Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase words; deliberately avoid external tokenizers and downloads."""
    return _TOKEN_RE.findall(text.lower())


def _canonical(token: str) -> str:
    return ALIASES.get(token, token)


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float


class Retriever:
    """Index a small corpus with deterministic scoring and explicit tie breaks.

    ``dense`` is a fixed feature-space baseline, not a learned embedding model.
    Its value is pedagogical: learners can inspect aliases and reproduce every score.
    """

    METHODS = frozenset({"tfidf", "dense", "hybrid"})

    def __init__(self, documents: Iterable[Document]):
        self.documents = tuple(documents)
        ids = [doc.doc_id for doc in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("document ids must be unique")
        self._by_id = {doc.doc_id: doc for doc in self.documents}
        self._tokens = {doc.doc_id: tokenize(f"{doc.title} {doc.text}") for doc in self.documents}
        self._canonical_tokens = {
            doc.doc_id: [_canonical(token) for token in self._tokens[doc.doc_id]]
            for doc in self.documents
        }
        self._vocabulary = sorted(
            {token for tokens in self._tokens.values() for token in tokens}
            | {token for tokens in self._canonical_tokens.values() for token in tokens}
        )
        self._index = {token: position for position, token in enumerate(self._vocabulary)}
        document_frequency = Counter(
            token for tokens in self._tokens.values() for token in set(tokens)
        )
        count = len(self.documents)
        self._idf = {
            token: math.log((1 + count) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }

    def _tfidf_vector(self, text: str) -> np.ndarray:
        vector = np.zeros(len(self._vocabulary), dtype=np.float64)
        tokens = tokenize(text)
        counts = Counter(tokens)
        for token, frequency in counts.items():
            if token in self._idf:
                vector[self._index[token]] = (1 + math.log(frequency)) * self._idf[token]
        return vector

    def _dense_vector(self, text: str) -> np.ndarray:
        vector = np.zeros(len(self._vocabulary), dtype=np.float64)
        counts = Counter(_canonical(token) for token in tokenize(text))
        for token, frequency in counts.items():
            if token in self._index:
                vector[self._index[token]] = 1 + math.log(frequency)
        return vector

    @staticmethod
    def _cosine(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return float(np.dot(left, right) / denominator) if denominator else 0.0

    def _score(self, query: str, doc: Document, method: str) -> float:
        if method == "tfidf":
            return self._cosine(
                self._tfidf_vector(query), self._tfidf_vector(f"{doc.title} {doc.text}")
            )
        if method == "dense":
            return self._cosine(
                self._dense_vector(query), self._dense_vector(f"{doc.title} {doc.text}")
            )
        content = f"{doc.title} {doc.text}"
        tfidf = self._cosine(self._tfidf_vector(query), self._tfidf_vector(content))
        dense = self._cosine(self._dense_vector(query), self._dense_vector(content))
        return 0.5 * tfidf + 0.5 * dense

    def search(self, query: str, k: int = 5, method: str = "hybrid") -> list[Hit]:
        if k <= 0:
            raise ValueError("k must be positive")
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {sorted(self.METHODS)}")
        if not tokenize(query) or not self.documents:
            return []
        scored = [Hit(doc.doc_id, self._score(query, doc, method)) for doc in self.documents]
        # All zero-score documents are omitted: retrieval should not fabricate evidence.
        scored = [hit for hit in scored if hit.score > 0 and math.isfinite(hit.score)]
        return sorted(scored, key=lambda hit: (-hit.score, hit.doc_id))[:k]

    def document(self, doc_id: str) -> Document:
        try:
            return self._by_id[doc_id]
        except KeyError as error:
            raise ValueError(f"unknown document: {doc_id}") from error

    def token_vector(self, token: str) -> np.ndarray:
        """Return a deterministic one-hot-ish vector for late interaction."""
        vector = self._dense_vector(_canonical(token))
        return vector


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]], rank_constant: float = 60.0
) -> list[Hit]:
    """Fuse ranked lists without comparing their incompatible score scales."""
    if rank_constant < 0 or not math.isfinite(rank_constant):
        raise ValueError("rank_constant must be finite and nonnegative")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, hit in enumerate(ranking):
            if hit.doc_id in seen:
                continue
            seen.add(hit.doc_id)
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1 / (rank_constant + rank + 1)
    return [
        Hit(doc_id, score)
        for doc_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def rerank(
    query: str,
    candidates: Sequence[Hit],
    k: int = 5,
    documents: Iterable[Document] = DOCUMENTS,
) -> list[Hit]:
    """Rescore query-document pairs by token coverage and adjacent phrase matches.

    This rule-based reranker is not a trained cross-encoder. Unlike first-stage
    scoring, it checks query bigrams in candidate text and ignores old scores.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len({hit.doc_id for hit in candidates}) != len(candidates):
        raise ValueError("candidates must have unique document ids")
    by_id = {document.doc_id: document for document in documents}
    query_tokens = [_canonical(token) for token in tokenize(query)]
    if not query_tokens:
        return []
    query_words = set(query_tokens)
    query_bigrams = set(itertools.pairwise(query_tokens))
    results = []
    for candidate in candidates:
        if candidate.doc_id not in by_id:
            raise ValueError(f"unknown candidate: {candidate.doc_id}")
        document = by_id[candidate.doc_id]
        tokens = [_canonical(token) for token in tokenize(f"{document.title} {document.text}")]
        coverage = len(query_words & set(tokens)) / len(query_words)
        bigrams = set(itertools.pairwise(tokens))
        phrase_bonus = len(query_bigrams & bigrams) / max(len(query_bigrams), 1)
        score = coverage + 0.25 * phrase_bonus
        results.append(Hit(candidate.doc_id, score))
    return sorted(results, key=lambda hit: (-hit.score, hit.doc_id))[:k]
