# Challenge (3+ hrs)

Implement a **cosine-similarity nearest-neighbor** function that:

1. Given a query batch (B, d) and a corpus (N, d), returns top-k indices per query.
2. Works on N up to 1M and d up to 768 without OOM on 16GB M-series.
3. Includes chunking: process the corpus in blocks of 100k rows to bound memory.
4. Runs in NumPy (baseline) AND PyTorch (with `torch.mm` and no autograd graph).
5. Timed on synthetic data (B=32, d=768, N∈{1e4, 1e5, 1e6}). Report throughput.

Stretch: quantize the corpus to int8 and re-time. Expect 2-4× speedup + 4× smaller memory.
