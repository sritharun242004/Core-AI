# COMPUTE — numpy-vs-pytorch

- **Tier:** 🟢 local M-series or any CPU.
- **Time:** `uv sync --extra dev` ≈ 60-90 s (torch download). `pytest` ≈ 10 s (torch imports dominate).
- **Budget:** $0. Torch runs on CPU/MPS on M-series; no CUDA needed.
