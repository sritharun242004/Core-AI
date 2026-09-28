# Warmup (30 min)

1. Time `matmul_python` vs `matmul_numpy` vs `matmul_torch` on 200×200 matrices. Report the ratio (expect: Python ≈ 100-1000× slower than NumPy; NumPy within 2× of Torch on CPU).
2. Verify `softmax_python([1000, 1001, 1002])` does not overflow. Where's the guard?
3. Compare `cross_entropy_numpy` output on a batch of size 1 vs `-log(softmax(logits)[label])` — should match to ~1e-9.
