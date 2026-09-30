# Compute note — Week 13

**Tier: 🟡 yellow.** The reference tests and notebook are CPU/offline and cost
**$0**. A small M-series laptop is sufficient for the teaching fixture; the
yellow label is honest because sequence length, layer count, and matched
attention-vs-SSM ablations can grow quickly, and MPS behavior differs by
PyTorch build.

## Reproducible local run

```bash
cd projects/week-13-nano-gpt-ssm
PYTHONPATH=src ../../.venv/bin/python -m pytest
../../.venv/bin/ruff check src tests
PYTHONPATH=src ../../.venv/bin/python notebooks/01-nano-gpt-ssm.py
```

The default suite uses tiny models, one deterministic in-memory fixture, and
no network or pretrained weights. It should take seconds to a few minutes on a
recent M-series Mac. Record Python/PyTorch versions, chip, device, seed,
`d_model`, heads/layers, SSM state width, sequence length, optimizer, learning
rate, batch size, and token/update budget.

## M-series budget (planning, not a promise)

For a serious local comparison, start with **20k–100k tokens**, sequence length
64–256, batch size chosen to fit unified memory, and 3–10 seeds only after one
seed is healthy. A tiny model (`d_model` 64–128, 2–4 layers) is generally a
reasonable M1/M2/M3/M4 CPU or MPS experiment. Budget **minutes to low hours**
for this scale, depending on token count, PyTorch version, thermal state, and
whether the explicit Python SSM loop or attention dominates. Do not present
these as measured benchmark timings until you record a run.

The reference comparison uses the same fixture and objective, not equal
parameter counts or optimized kernels. For a fair extension, match token
updates, seed set, vocabulary, context length, optimizer, learning-rate
schedule, evaluation windows, and parameter-count range. Report mean and spread
rather than the best seed.

## Optional larger path

A larger local corpus must be supplied explicitly and kept outside the default
tests. If using MPS, verify numerical behavior on a CPU smoke run first and
record `PYTORCH_ENABLE_MPS_FALLBACK` if it is set. A cloud GPU is optional, not
needed for the reference path; choose a fixed spend cap and stop the instance
manually. No cloud command, download command, or credential is included here.

## Resource boundary

The toy loss reduction is not a language-quality benchmark. The explicit SSM
loop is intentionally inspectable rather than kernel optimized, so timing it
against fused production kernels would be misleading. Attention probabilities
also expose an `O(T²)` score matrix, while the baseline stores an `O(Td)` state
per sequence step. Those are useful complexity observations, not claims of
end-to-end device speed.
