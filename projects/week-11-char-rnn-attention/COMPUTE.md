# Compute note

**Tier:** 🟢 local CPU · **tests:** 🟢 offline · **budget:** $0

The reference suite uses only PyTorch, a deterministic in-memory character
fixture, and small CPU tensors. It never imports a downloader, calls a data
service, or contacts the network. Ten tests should finish in seconds on a
recent laptop. MPS and CUDA are optional accelerators for an experiment, not a
requirement of the lesson.

## Reproducible local run

1. Run the isolated pytest and Ruff commands from `README.md`.
2. Keep the fixture in memory and set `torch.manual_seed` before constructing a
   model when comparing runs.
3. Record vocabulary size, sequence length, batch size, hidden and embedding
   widths, optimizer, learning rate, gradient clipping, epochs, device, and
   whether decoding used teacher forcing or free-running generation.
4. Treat the loss-reduction smoke test as a plumbing check. It is not a
   language-quality benchmark.

A small CPU configuration (sequence length 32, hidden width 48, batch size 32,
three epochs on the built-in fixture) is intentionally modest. Longer windows,
multiple layers, or a local MPS run can be useful for exploration but should
remain reproducible and should not be silently substituted into the tests.

## Optional local text

The reference project does **not** download Shakespeare. If you already have a
legally obtained local text file, read it explicitly and document its path,
encoding, preprocessing, train/validation split, and token count. Do not commit
that corpus or use a held-out file for tuning. A longer run should save a
checkpoint and a JSON manifest before comparing samples.

## Cloud boundary

No cloud provider, credentials, instance type, or spend is needed for Week 11.
Do not provision a GPU merely to pass this project. If you choose a larger
experiment, set an explicit budget, enable checkpointing, and tear down the
instance after the run; the offline reference implementation does not automate
cloud provisioning.
