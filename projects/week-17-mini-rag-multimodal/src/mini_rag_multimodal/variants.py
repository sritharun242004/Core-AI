"""Small, explainable retrieval variants: graph expansion, HyDE, and late interaction."""

import math

import numpy as np

from .retrieval import Hit, Retriever, tokenize


def graph_expand(index: Retriever, query: str, hops: int = 1, k: int = 5) -> list[Hit]:
    """Retrieve seeds, then walk document links for at most ``hops`` rounds."""
    if hops < 0:
        raise ValueError("hops must be nonnegative")
    if k <= 0:
        raise ValueError("k must be positive")
    seeds = index.search(query, k=k, method="hybrid")
    if not seeds:
        raise ValueError("query matched no documents")
    scores = {hit.doc_id: hit.score for hit in seeds}
    frontier = {hit.doc_id: hit.score for hit in seeds}
    visited = set(frontier)
    for hop in range(1, hops + 1):
        next_frontier: dict[str, float] = {}
        for doc_id, parent_score in sorted(frontier.items()):
            for linked_id in index.document(doc_id).links:
                propagated = parent_score / (hop + 1)
                scores[linked_id] = max(scores.get(linked_id, 0.0), propagated)
                if linked_id not in visited:
                    next_frontier[linked_id] = max(next_frontier.get(linked_id, 0.0), propagated)
        visited.update(next_frontier)
        frontier = next_frontier
    return [
        Hit(doc_id, score)
        for doc_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def hyde_query(query: str, hypothetical_answer: str, repeats: int = 1) -> str:
    """Concatenate a deterministic hypothetical answer; no language model is called."""
    if repeats < 0:
        raise ValueError("repeats must be nonnegative")
    pieces = [query.strip()] + [hypothetical_answer.strip()] * repeats
    return " ".join(piece for piece in pieces if piece)


def late_interaction_search(index: Retriever, query: str, k: int = 5) -> list[Hit]:
    """Toy ColBERT-like max-sim: each query token chooses its best doc token."""
    if k <= 0:
        raise ValueError("k must be positive")
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    query_vectors = [index.token_vector(token) for token in query_tokens]
    results: list[Hit] = []
    for document in index.documents:
        document_vectors = [index.token_vector(token) for token in tokenize(document.text)]
        if not document_vectors:
            continue
        score = sum(
            max(
                float(np.dot(query_vector, document_vector))
                / max(float(np.linalg.norm(query_vector) * np.linalg.norm(document_vector)), 1e-12)
                for document_vector in document_vectors
            )
            for query_vector in query_vectors
        ) / len(query_vectors)
        if score > 0 and math.isfinite(score):
            results.append(Hit(document.doc_id, score))
    return sorted(results, key=lambda hit: (-hit.score, hit.doc_id))[:k]
