# %% [markdown]
# Week 17 — inspect a mini-RAG and multimodal seams
#
# This percent-format notebook is offline and deterministic. It uses a tiny hand-written
# corpus instead of a downloaded dataset or pretrained checkpoint. Every score is a
# teaching trace, not evidence of model quality.

# %%
from mini_rag_multimodal import (
    DOCUMENTS,
    QRELS,
    QUERIES,
    Retriever,
    VoicePipeline,
    denoise_step,
    evaluate_rankings,
    graph_expand,
    hyde_query,
    late_interaction_search,
    tiny_vae_roundtrip,
)
from mini_rag_multimodal.retrieval import reciprocal_rank_fusion, rerank

index = Retriever(DOCUMENTS)
print("documents:", [document.doc_id for document in DOCUMENTS])
print("queries:", QUERIES)

# %% [markdown]
# ## 1. First-stage retrieval
#
# TF-IDF is lexical. The dense fixture is a fixed alias-expanded feature space, not a
# trained embedding model. Hybrid averages their cosine scores.

# %%
for query_id, query in QUERIES.items():
    print("\n", query_id, query)
    for method in ("tfidf", "dense", "hybrid"):
        hits = index.search(query, k=3, method=method)
        print(method, [(hit.doc_id, round(hit.score, 3)) for hit in hits])

# %% [markdown]
# ## 2. Retrieval metrics
#
# We macro-average four metrics over the labelled query set. NDCG uses graded labels;
# MAP divides by every relevant document, including relevant documents not retrieved.

# %%
for method in ("tfidf", "dense", "hybrid"):
    rankings = {
        query_id: [hit.doc_id for hit in index.search(query, k=3, method=method)]
        for query_id, query in QUERIES.items()
    }
    print(method, evaluate_rankings(rankings, QRELS, k=3))

# %% [markdown]
# ## 3. Candidate fusion and reranking
#
# Reciprocal-rank fusion combines rankings without pretending their raw cosine scales
# are comparable. Reranking is candidate preserving: it may reorder candidates but cannot
# invent evidence.

# %%
query = QUERIES["q_diffusion"]
tfidf_hits = index.search(query, k=4, method="tfidf")
dense_hits = index.search(query, k=4, method="dense")
fused = reciprocal_rank_fusion([tfidf_hits, dense_hits])
print("fused:", [(hit.doc_id, round(hit.score, 3)) for hit in fused])
print("reranked:", rerank(query, fused, k=3))

# %% [markdown]
# ## 4. GraphRAG expansion, HyDE, and late interaction
#
# The graph is explicit links in the fixture. HyDE here is a deterministic query seam;
# no model writes the hypothetical answer. Late interaction computes a max similarity
# for each query token over document tokens and averages those maxima.

# %%
print("graph expansion:", graph_expand(index, "clip", hops=1, k=1))
expanded = hyde_query("find a spoken interface", "audio transcription retrieval", repeats=1)
print("HyDE query:", expanded)
print("HyDE retrieval:", index.search(expanded, k=3, method="dense"))
print("late interaction:", late_interaction_search(index, "denoise noisy image", k=3))

# %% [markdown]
# ## 5. Tiny VAE and one denoising step
#
# These NumPy functions show latent shapes, reparameterization, ELBO terms, seeded
# corruption, and a finite denoising update. They are not trained image generators.

# %%
image = __import__("numpy").array([[0.0, 0.25], [0.75, 1.0]])
vae = tiny_vae_roundtrip(image, latent_size=2, seed=4)
print("latent:", vae.latent)
print("reconstruction shape:", vae.reconstruction.shape)
print("reconstruction loss / KL / ELBO:", vae.reconstruction_loss, vae.kl, vae.loss)
denoised = denoise_step(image, noise_scale=0.2, seed=4)
print("denoising losses:", denoised.loss_before, denoised.loss_after)
print("finite step:", __import__("numpy").isfinite(denoised.restored).all())

# %% [markdown]
# ## 6. Voice-native pipeline seam
#
# The offline transcriber decodes UTF-8 fixture bytes; the offline synthesizer returns
# tagged bytes. In a real local deployment, replace those adapters while preserving the
# transcript, citation, no-evidence, and audio contracts.

# %%
voice = VoicePipeline(index).run(b"spoken audio pipeline", k=2)
print("transcript:", voice.transcript)
print("hits:", voice.hits)
print("response:", voice.response)
print("audio bytes:", voice.audio)

# %% [markdown]
# ## Reflection
#
# Retrieval metrics measure ranking against labels, not factuality. Graph expansion can
# improve recall while adding irrelevant context. HyDE can improve lexical overlap while
# hallucinating a query expansion. A VAE/diffusion-shaped tensor trace proves finite,
# reproducible operations, not image quality. Voice interfaces make replacement points
# explicit, not speech recognition.
