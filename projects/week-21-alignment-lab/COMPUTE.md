# Compute — Week 21 alignment lab

**Tier: 🟡 yellow.** The course topic permits optional local experiments, but this reference implementation is intentionally tiny and CPU-only. No GPU or paid service is required.

| Workload | Default size | Planning budget |
|---|---|---|
| DPO | 312 policy parameters + frozen copy; 8 pairs; 100 full-batch steps | Seconds on a modern laptop |
| Synthetic safety battery | 20 prompts; at most 2 mock calls per case | Below a second, in memory |
| Linear probe | 8 inputs; 256 train / 128 test; 200 steps | Seconds including framework startup |
| SAE | 8 inputs / 12 features; 192 train / 96 test; 250 steps | Seconds including framework startup |

Allow **under one minute** for the tests and full notebook on a typical laptop; this is a planning estimate, not a hardware benchmark. The implementation validation run had 27 tests pass in about 1.3 seconds excluding initial interpreter/framework overhead. Tensor storage is tiny; installed PyTorch and its process overhead dominate memory. A machine with a few GB of available RAM is sufficient for this fixture; no frontier-model memory claim is implied.

- **Dollar budget:** $0 external compute/API spend; ordinary local electricity and existing software storage only.
- **Network/data:** no dataset, tokenizer or model downloads. Fixtures are hand-written or seeded in memory.
- **Environment:** use the existing root `.venv/bin/python` with project `src` on `PYTHONPATH`; do not change shared lockfiles or run workspace synchronization to execute the lab.
- **Reproducibility:** default seed 21, float32 CPU, one Torch thread in tests/notebook. Exact last digits can vary across Torch/platform versions. The tests assert numerical identities and meaningful improvement bounds, not a marketing benchmark.
- **Teardown:** scripts exit by themselves. No server, worker queue, tracking service or cloud resource is created. The notebook does not write model checkpoints or output artifacts.

## Optional scale-up, not run here

If you later reuse a locally available text checkpoint, define a separate experiment: document checkpoint/tokenizer licenses, completion masks, data consent, train/reference memory, sequence lengths, effective batch size and a wall-time ceiling before starting. Keep the reference frozen and restore Week 15a's multi-token scoring tests. Use a private heldout suite with benign controls, not these development prompts.

Do not extrapolate this toy's seconds, quality or cost to a 1B-parameter LM or an SAE trained on millions of activations. GPU/cloud jobs, provider API evaluations, causal steering and full Anthropic Circuits reproductions are deliberately outside this lab. No cloud runbook is needed for the default yellow-tier deliverable because it creates no remote resources.
