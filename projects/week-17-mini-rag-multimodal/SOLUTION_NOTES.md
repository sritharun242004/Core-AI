# Solution notes

Keep the distinctions below visible in a report:

- **First-stage retrieval:** TF-IDF counts literal terms with inverse-document
  frequency. The dense fixture maps a small, listed alias set into the same inspectable
  vector space. Hybrid averages the two cosine scores; in a real system, dense vectors
  usually come from a separately trained encoder.
- **Reranking:** the lab scores only candidates, based on query token coverage and
  matching bigrams. It never invents a document. A cross-encoder would jointly encode a
  query/document pair and is an optional extension, not implemented here.
- **Metrics:** Recall divides by all positive relevance labels; MRR uses the first hit;
  MAP averages precision at each positive and divides by all positives; NDCG uses
  exponential graded gain and an ideal ranking. Missing qrels are not silently removed
  from the macro denominator.
- **GraphRAG:** the document links are a toy knowledge graph. Expansion propagates
  seed score with a hop penalty and has a bounded visited set. It is not graph
  extraction, entity resolution, or a claim about a production GraphRAG system.
- **HyDE:** `hyde_query` concatenates a supplied hypothetical answer. It stands in for
  the query-expansion seam; there is no language-model generation or factual guarantee.
- **Late interaction:** query tokens independently take their maximum cosine with
  document tokens, then the maxima are averaged. This is the pedagogical max-sim idea
  behind ColBERT, not a ColBERT checkpoint or tokenizer.
- **Generative fixture:** the VAE reports a mean, log variance, sampled latent,
  reconstruction MSE, and KL term. The denoising function adds seeded Gaussian noise
  and performs a local mean update. Neither has been trained on images.
- **Voice:** offline transcriber decodes fixture text bytes and offline synthesizer
  emits tagged bytes. Injection points make it possible to test an eventual local ASR or
  TTS adapter without changing retrieval contracts.

A useful extension report includes corpus version, query/qrel version, method, k,
reranker candidate count, seed, latency, metrics, and at least one failure case. Avoid
reporting a single retrieved answer as if it established factual correctness.
