# COMPUTE — micrograd

- **Tier:** 🟢 local M-series (or any Python 3.13 machine).
- **Time:** `uv sync` ≈ 30 s; `pytest` ≈ 1 s; XOR notebook trains in < 5 s.
- **Budget:** $0. No GPU, no cloud.
- **`torch` install:** optional; the parity test uses `importorskip`. Skip it if `pip install torch` is heavyweight for your platform.
