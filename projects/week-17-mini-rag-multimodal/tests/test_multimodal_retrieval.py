import numpy as np
import pytest
from mini_rag_multimodal.corpus import DOCUMENTS, QRELS, QUERIES, Document
from mini_rag_multimodal.retrieval import Retriever, reciprocal_rank_fusion, rerank


@pytest.mark.parametrize("method", ["tfidf", "dense", "hybrid"])
def test_known_query_order_is_deterministic(method: str) -> None:
    index = Retriever(DOCUMENTS)
    first = index.search("diffusion noise denoising", k=3, method=method)
    second = index.search("diffusion noise denoising", k=3, method=method)
    assert first == second
    assert first[0].doc_id == "diffusion"
    assert all(np.isfinite(hit.score) for hit in first)
    assert len({hit.doc_id for hit in first}) == len(first)


def test_dense_concept_synonym_can_retrieve_without_literal_word_overlap() -> None:
    index = Retriever(DOCUMENTS)
    assert index.search("utterances", method="tfidf") == []
    assert index.search("utterances", method="dense")[0].doc_id == "voice"


def test_unknown_or_empty_queries_do_not_create_fake_evidence() -> None:
    index = Retriever(DOCUMENTS)
    for method in ("tfidf", "dense", "hybrid"):
        assert index.search("", method=method) == []
        assert index.search("zyxwvu", method=method) == []


def test_ties_break_by_document_id_not_input_order() -> None:
    docs = (Document("z", "same", "token"), Document("a", "same", "token"))
    hits = Retriever(docs).search("token", method="tfidf")
    assert [hit.doc_id for hit in hits] == ["a", "z"]


def test_rrf_uses_rank_instead_of_incompatible_score_scales() -> None:
    index = Retriever(DOCUMENTS)
    a = index.search("diffusion", k=1, method="tfidf")[0]
    b = index.search("voice", k=1, method="tfidf")[0]
    fused = reciprocal_rank_fusion([[a, b], [b]], rank_constant=0)
    assert [hit.doc_id for hit in fused] == ["voice", "diffusion"]
    assert fused[0].score == 1.5
    assert fused[1].score == 1.0


def test_rerank_orders_candidates_and_never_invents_a_candidate() -> None:
    index = Retriever(DOCUMENTS)
    candidates = index.search("noise image", k=8, method="hybrid")
    reranked = rerank("diffusion noise denoising", candidates, k=3)
    assert reranked[0].doc_id == "diffusion"
    assert {hit.doc_id for hit in reranked} <= {hit.doc_id for hit in candidates}
    assert reranked == rerank("diffusion noise denoising", candidates, k=3)


def test_index_rejects_duplicate_ids_and_unknown_methods() -> None:
    with pytest.raises(ValueError, match="unique"):
        Retriever((DOCUMENTS[0], DOCUMENTS[0]))
    with pytest.raises(ValueError, match="method"):
        Retriever(DOCUMENTS).search("image", method="magic")
    with pytest.raises(ValueError, match="positive"):
        Retriever(DOCUMENTS).search("image", k=0)


def test_empty_corpus_and_fixture_qrels_are_well_defined() -> None:
    assert Retriever(()).search("diffusion") == []
    ids = {document.doc_id for document in DOCUMENTS}
    assert set(QUERIES) == set(QRELS)
    assert all(set(labels) <= ids for labels in QRELS.values())
