import numpy as np
import pytest
from mini_rag_multimodal.corpus import DOCUMENTS
from mini_rag_multimodal.generation import denoise_step, tiny_vae_roundtrip
from mini_rag_multimodal.retrieval import Retriever
from mini_rag_multimodal.variants import graph_expand, hyde_query, late_interaction_search


def test_graph_expansion_is_bounded_and_follows_known_edges() -> None:
    index = Retriever(DOCUMENTS)
    expanded = graph_expand(index, "clip", hops=1, k=1)
    assert expanded[0].doc_id == "clip"
    assert {hit.doc_id for hit in expanded} >= {"clip", "vit"}
    assert len(expanded) <= len(DOCUMENTS)
    assert graph_expand(index, "clip", hops=0, k=1)[0].doc_id == "clip"


def test_graph_rejects_negative_hops_and_unknown_seed() -> None:
    index = Retriever(DOCUMENTS)
    with pytest.raises(ValueError, match="hops"):
        graph_expand(index, "clip", hops=-1)
    with pytest.raises(ValueError, match="no documents"):
        graph_expand(index, "zzzzzz", hops=1)


def test_hyde_is_deterministic_and_expands_a_query_without_network() -> None:
    expanded = hyde_query("find images", "dense visual embeddings", repeats=2)
    assert expanded == "find images dense visual embeddings dense visual embeddings"
    assert hyde_query("find images", "dense visual embeddings", repeats=0) == "find images"
    with pytest.raises(ValueError, match="repeats"):
        hyde_query("find images", "answer", repeats=-1)


def test_late_interaction_max_sim_prefers_token_alignment() -> None:
    index = Retriever(DOCUMENTS)
    hits = late_interaction_search(index, "denoise noisy image", k=4)
    assert hits[0].doc_id == "diffusion"
    assert hits == late_interaction_search(index, "denoise noisy image", k=4)
    assert all(np.isfinite(hit.score) for hit in hits)


def test_late_interaction_has_empty_query_contract() -> None:
    assert late_interaction_search(Retriever(DOCUMENTS), "", k=3) == []
    with pytest.raises(ValueError, match="positive"):
        late_interaction_search(Retriever(DOCUMENTS), "image", k=0)


def test_tiny_vae_and_denoising_are_finite_and_bounded() -> None:
    image = np.array([[0.0, 0.25], [0.75, 1.0]], dtype=np.float64)
    result = tiny_vae_roundtrip(image, latent_size=2, seed=4)
    assert result.latent.shape == (2,)
    assert result.reconstruction.shape == image.shape
    assert np.isfinite(result.latent).all()
    assert np.isfinite(result.reconstruction).all()
    assert 0 <= result.reconstruction.min() <= result.reconstruction.max() <= 1
    clean, noisy, restored = denoise_step(image, noise_scale=0.2, seed=4)
    assert np.array_equal(clean, image)
    assert np.isfinite(noisy).all() and np.isfinite(restored).all()
    assert restored.shape == image.shape


def test_generative_step_rejects_bad_noise_scale() -> None:
    with pytest.raises(ValueError, match="noise_scale"):
        denoise_step(np.zeros((2, 2)), noise_scale=-0.1)
