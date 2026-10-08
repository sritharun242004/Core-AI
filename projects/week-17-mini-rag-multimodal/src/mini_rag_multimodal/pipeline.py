# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""End-to-end retrieve-then-answer seam for the notebook and voice adapter."""

from dataclasses import dataclass

from .retrieval import Hit, Retriever, rerank


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[str, ...]
    hits: tuple[Hit, ...]


def answer_question(retriever: Retriever, query: str, k: int = 3) -> Answer:
    """Return an extractive, citation-bearing answer without an LLM or network."""
    if k <= 0:
        raise ValueError("k must be positive")
    candidates = retriever.search(query, k=max(k, 5), method="hybrid")
    hits = tuple(rerank(query, candidates, k=k, documents=retriever.documents))
    if not hits:
        return Answer("I found no local evidence for that question.", (), ())
    snippets = []
    for hit in hits:
        document = retriever.document(hit.doc_id)
        snippets.append(f"[{document.doc_id}] {document.text}")
    return Answer(
        text="Offline evidence: " + " ".join(snippets),
        citations=tuple(hit.doc_id for hit in hits),
        hits=hits,
    )
