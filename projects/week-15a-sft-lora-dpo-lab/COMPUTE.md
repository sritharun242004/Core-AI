# Compute note — Week 15a

**Tier: 🔴 red.** The required reference project is an offline CPU toy and
costs **$0**. Red describes an optional larger post-training extension, not a
requirement for the tests or notebook. Do not run cloud commands from the
default path. Pass the local contracts first, then specify a checkpoint,
data license, evaluation set, and spend cap before provisioning a GPU.

## Local toy path (default and required)

```bash
cd projects/week-15a-sft-lora-dpo-lab
PYTHONPATH=src ../../.venv/bin/python -m pytest
../../.venv/bin/ruff check src tests notebooks
PYTHONPATH=src ../../.venv/bin/python notebooks/01-sft-lora-dpo.py
```

This path uses a tiny decoder, in-memory token ids, deterministic objectives,
and ordinary PyTorch. It needs no GPU, model download, API key, cloud SDK,
quantization package, or external service.

## Optional A100 40 GB planning envelope

The following is a planning estimate, not a measured benchmark or a provider
quote. It reserves **2 A100 40 GB GPU-hours** for a bounded adapter experiment:
load a named local/open checkpoint, run a small licensed SFT or preference
slice, evaluate a fixed prompt set, and save only metrics. Provider prices
change with region, spot status, storage, egress, minimum billing, and image
startup time; confirm the current quote immediately before provisioning.

| Provider | Estimated time | Planning rate | Estimated GPU cost | Teardown |
| --- | ---: | ---: | ---: | --- |
| RunPod A100 40 GB | 2 GPU-hours | $2–$5/hour | **$4–$10**, plus storage | Stop/delete the pod and any attached network volume in the dashboard; verify there is no running pod or retained volume. |
| Modal A100 40 GB | 2 GPU-hours | $2.50–$5/hour | **$5–$10**, plus image/volume time | End the function/job, then inspect the Modal dashboard for active apps, volumes, schedules, and images; remove unused resources. |
| Vast.ai A100 40 GB | 2 GPU-hours | $1.50–$4/hour | **$3–$8**, plus host/storage/network | Stop and destroy the instance from the Vast dashboard; verify no rental, disk, or reservation remains. |

Use a hard cap of **$10 per provider** and a two-hour wall-clock timeout for
this planning baseline. An A100 40 GB is an optional capacity reference, not
evidence that this toy needs one. A small LoRA run may fit on less hardware;
a long-sequence checkpoint, optimizer state, or quantization implementation
can change the memory budget.

Before a hosted run, record:

- checkpoint name, revision, license, tokenizer, and whether base weights are
  trusted;
- dataset source, license, prompt/response filtering, and PII policy;
- seed, sequence length, batch/accumulation, precision, rank/alpha/dropout;
- SFT/DPO objective, beta/gamma, reference checkpoint, update count, and
  held-out evaluation prompts;
- expected GPU-hours, maximum spend, retention period, and experiment id; and
- chosen/rejected win rate, loss, KL/reference gap, peak memory, throughput,
  latency, and failure/safety slices.

## Teardown checklist

1. Stop the training job only after the final checkpoint/metrics flush.
2. Stop and delete the GPU instance or job at the provider; closing a terminal
   does not stop billing.
3. Delete or detach persistent volumes, snapshots, idle endpoints, scheduled
   jobs, and unused images created by the experiment.
4. Check usage/billing and record actual GPU-hours and cost against the cap.
5. Remove temporary credentials and rotate any credential that was printed or
   written to logs.

No provider CLI command is included intentionally: syntax and billing
semantics change, and a copied destructive command can target the wrong
resource. Use the current provider dashboard/docs and confirm the exact
resource id before deletion.

## Honest boundary

A two-hour A100 budget can support one carefully bounded adapter smoke test. It
cannot establish a scaling law, a production QLoRA speedup, preference
robustness, alignment, or a claim about a private company's training stack.
The local evidence is narrower: finite objective tensors, identity-preserving
LoRA injection, a decreasing fixture loss, and exact checkpoint restoration.
