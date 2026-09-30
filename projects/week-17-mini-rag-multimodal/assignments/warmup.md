# Warmup — read the retrieval traces (30–45 min)

1. Open `corpus.py`. Write down the eight document ids, three graph links, and the
   graded qrels for `q_visual`.
2. For `q_diffusion`, compute by hand which term a TF-IDF vector shares with
   `diffusion` and why an all-zero query must return no evidence.
3. Compare `tfidf`, `dense`, and `hybrid` for one query. Explain why the dense fixture
   can retrieve `voice` for `utterances` even though the literal token is absent.
4. Given ranking `[x, a, b]` and relevance `{a: 1, b: 1, c: 1}`, calculate Recall@2,
   MRR@3, and AP@3. Check your work against `metrics.py`.
5. Trace one hop from `clip` in the links. Why should a hop penalty exist? What would
   be a failure mode if expansion were unbounded?
6. Explain max-sim in `late_interaction_search`: what does each query token maximize
   over, and why is averaging those maxima different from one pooled vector?
7. Run the voice pipeline with bytes and inspect the tagged output. Identify which
   method would be replaced by a real local ASR adapter and which by a TTS adapter.

Start with a failing assertion if you change a behavior. No network or pretrained
model is needed.
