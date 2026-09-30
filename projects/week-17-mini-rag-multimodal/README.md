# Week 17 — mini-RAG + multimodal foundations

An offline-first, CPU-friendly lab for reasoning about retrieval and multimodal system
boundaries. It uses eight hand-written documents, fixed token features, and seeded NumPy
operations. There are no downloads, model checkpoints, API calls, or required optional
packages.

## What is implemented

- deterministic corpus with explicit links and graded query relevance labels;
- TF-IDF cosine retrieval, a transparent dense feature-space baseline, and a hybrid;
- candidate-preserving rule-based reranking and reciprocal-rank fusion;
- Recall@k, MRR, NDCG@k, and MAP@k with documented zero/missing-query behavior;
- graph expansion over document links, deterministic HyDE query concatenation, and a
  ColBERT-like late-interaction max-sim score;
- a tiny NumPy VAE round trip with reparameterization/ELBO terms and a one-step local
  denoising demonstration (a diffusion-shaped teaching fixture, not a trained model);
- injectable voice-native interfaces: offline UTF-8 ASR/TTS adapters make the
  transcribe → retrieve → answer → synthesize seam executable without audio APIs.

The dense baseline is intentionally **not** a pretrained embedding model, the reranker
is not a cross-encoder, the graph is not an enterprise knowledge graph, and the voice
adapters do not perform speech recognition. Those boundaries are the point of the lab:
replace one seam at a time and keep evidence about what actually ran.

## Run locally

From the repository root:

```bash
PYTHONPATH=projects/week-17-mini-rag-multimodal/src \
  .venv/bin/pytest -q projects/week-17-mini-rag-multimodal/tests
PYTHONPATH=projects/week-17-mini-rag-multimodal/src \
  .venv/bin/python projects/week-17-mini-rag-multimodal/notebooks/01-mini-rag-multimodal.py
.venv/bin/ruff check projects/week-17-mini-rag-multimodal
```

`numpy` is the only runtime dependency. See `COMPUTE.md` for the yellow-tier budget
and optional cloud guidance. Read `assignments/` in order, then inspect
`SOLUTION_NOTES.md` only after attempting the work.
