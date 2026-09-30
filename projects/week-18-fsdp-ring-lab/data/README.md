# Data provenance

No downloaded or private data. Tests and notebook construct seeded PyTorch tensors in
memory; the GPU smoke seeds each rank/step separately (`1800 + 10000*step + rank`).
All examples are synthetic arithmetic fixtures, not a language-training benchmark.
Padding is transport metadata and never contributes an example or valid target token.
