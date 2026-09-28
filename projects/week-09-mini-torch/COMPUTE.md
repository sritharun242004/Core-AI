# Compute note

**Tier:** 🟢 local CPU · **Expected time:** under 2 minutes for tests and notebook · **Budget:** $0

Week 9 is intentionally runnable on an Apple Silicon laptop CPU (or any recent
Python machine). The reference implementation uses small float64 NumPy arrays;
there is no GPU requirement, cloud provider, credential, dataset download, or
network request. The built-in fixture is synthetic seven-segment data, not a
claim that real MNIST training has been reproduced.

The default tests train only tiny tables and a 100-row fixture. For a slower
experiment, increase the fixture size or hidden width in the notebook, but keep
an eye on the quadratic cost of building Python-level autodiff graphs. A real
framework uses fused kernels and should be used for serious workloads.
