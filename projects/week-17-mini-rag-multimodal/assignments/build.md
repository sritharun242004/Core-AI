# Build — evaluate a grounded answer (2–3 hrs)

1. Run the notebook and collect rankings for every query with TF-IDF, dense, and
   hybrid retrieval. Report Recall@3, MRR@3, NDCG@3, and MAP@3 for each method.
2. Add a candidate budget experiment: retrieve five documents, rerank, and return two.
   Test that the reranker never returns a document outside its candidate set.
3. Compare one-hop graph expansion and deterministic HyDE. State which new documents
   each method introduces and whether the qrels say those documents are relevant.
4. Add a failure case with an unknown query. The correct behavior is an explicit “no
   local evidence” result, not a plausible-sounding unsupported answer.
5. Run the VAE and denoising demonstration twice with the same seed. Record tensor
   shapes, finite checks, reconstruction/KL terms, and the denoising losses.
6. Inject a fake transcriber and synthesizer. Keep the interfaces stable and add a test
   that proves no network socket is needed.

Your short report must include corpus/query versions, seed, k, method, candidate budget,
all four metrics, one failure slice, and the boundary between this rule-based lab and a
trained model. Do not call the fixed dense feature map “semantic understanding.”
