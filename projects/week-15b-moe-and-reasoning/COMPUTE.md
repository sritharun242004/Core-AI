# Compute note — Week 15b

**Tier: 🔴 red.** The reference project is deliberately an offline CPU toy and
costs **$0**. The red tier is for an optional, reproducible larger MoE/reasoning
experiment, not for the tests. Do not run cloud commands from the test suite,
notebook, or default lesson path. A cloud run is useful only after the local
contracts pass and a dataset, checkpoint, and evaluation protocol have been
specified.

## Local toy path (default and required for verification)

```bash
cd projects/week-15b-moe-and-reasoning
PYTHONPATH=src ../../.venv/bin/python -m pytest
../../.venv/bin/ruff check src tests
PYTHONPATH=src ../../.venv/bin/python notebooks/01-moe-reasoning.py
```

This path uses tiny in-memory tensors, deterministic arithmetic, and PyTorch.
It needs no GPU, credential, model download, cloud account, or external
service. Record Python/PyTorch versions, CPU, seed, number of experts, top-k,
capacity factor, batch/time/model widths, optimizer, learning rate, update
count, and exact verifier before comparing runs.

## Planning baseline: one A100 40 GB

The following is a **budget envelope, not a measured benchmark or quote**. A
single A100 40 GB, with a small-to-medium open checkpoint and a short fixed
experiment, is a useful baseline for comparing dense versus sparse routing or
running a limited verifiable-reward pilot:

| Provider | Reservation for baseline | Planning rate | Budget envelope | Teardown |
|---|---:|---:|---:|---|
| RunPod A100 40 GB | 2 GPU-hours | $2–$5/hour | **$4–$10** plus storage | Stop/delete the pod and attached volume in the RunPod console; verify the pod status is stopped/deleted and release any network volume or template. |
| Modal A100 40 GB | 2 GPU-hours | $2.50–$5/hour | **$5–$10** plus image/storage | End the function/job, then check the Modal dashboard for active apps, volumes, and scheduled jobs; remove unused volumes/images. Modal billing depends on actual seconds and storage. |
| Vast.ai A100 40 GB | 2 GPU-hours | $1.50–$4/hour | **$3–$8** plus host/storage/network | Stop and destroy the instance from the Vast dashboard/API; verify no instance, storage, or reserved rental remains. Host prices and reliability vary. |

Set a hard spend cap of **$10 per provider** for this baseline, use a two-hour
wall-clock timeout, and do not leave a GPU running while inspecting results.
Prices change by region, host, spot status, storage, egress, and billing
minimums; confirm the current provider quote immediately before provisioning.
An A100 40 GB is a capacity baseline, not a claim that this toy lab needs one.

### What to measure if the baseline is approved

Use the same seed, token/update budget, precision, sequence length, batch
shape, checkpoint, data slice, expert count, top-k, capacity factor, and
router-drop policy. Log wall-clock time, peak allocated/reserved memory, tokens
per second, dropped-assignment rate, auxiliary loss, task reward, verifier
pass rate, and majority-vote pass rate. Compare dense and sparse systems at a
matched quality target; do not infer a speedup from parameter count alone.

Before starting, write down the dataset license, checkpoint provenance,
experiment ID, expected GPU-hours, maximum spend, and retention policy. Save
only the small metrics/config artifact needed for the lesson. Never put tokens,
credentials, private prompts, or user data in this repository.

## Teardown checklist

1. Stop the training process and wait for the final metrics/checkpoint flush.
2. Stop/delete the GPU instance or job at the provider, not merely the shell.
3. Delete or detach persistent volumes, snapshots, idle endpoints, and unused
   images that were created for this run.
4. Check the provider billing/usage page and record actual GPU-hours and cost.
5. Remove temporary credentials from the shell and rotate a credential if it
   was accidentally printed or saved.

No provider command is included intentionally: CLI syntax and billing semantics
change, and a pasted command can destroy unrelated resources. Use the named
provider's current dashboard/documentation, select the exact A100 40 GB
resource, and confirm the resource ID before deletion.

## Honest boundary

A 2-hour A100 budget can support a carefully bounded teaching run; it cannot
establish a scaling law, production MoE efficiency, or a claim about a private
company's model. The local tests establish routing sums/top-k validity, expert
shapes, finite load balancing, verifier behavior, grouped zero-mean advantages,
and a toy loss decrease. Report uncertainty and failed capacity cases instead
of hiding dropped tokens or selecting only the best seed.
