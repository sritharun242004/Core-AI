# Compute contract — red tier, with a free correctness path

## Required: $0 offline

Python 3.13 + NumPy + pytest on CPU (M-series/Linux/Windows). Tests and the percent
notebook need no torch, Transformers, vLLM, network, weights, cloud account, or API
key. Allow <1 minute and comfortably under 1 GB RAM for the fixtures. This is an
estimate, not a memory profiler result. `demo` explicitly marks simulated timings.

## Optional cloud plan (not executed by this project)

**Provider:** RunPod on-demand pod. **Baseline:** one NVIDIA A100 40 GB, Linux x86_64,
with a matching CUDA/PyTorch/vLLM image. One GPU is enough for this serial experiment;
start with an already locally available 0.5–3B decoder model, not a 70B checkpoint.
The model must fit with KV cache, runtime workspaces, and a safety margin. A100 supply,
regional quotas, and Python 3.13 wheel support vary. Modal or vast.ai are alternatives,
but rewrite the lifecycle checklist for that provider rather than assuming parity.

**Planning allowance:** 1–2 hours at an assumed **$1–3/GPU-hour**, or **$1–6 compute**;
reserve **$10 maximum** including storage/transfer. These are planning assumptions,
not current provider quotes. Read the live price, billing interval, minimum charge,
volume and egress rates before accepting. If the actual quote exceeds the cap, stop
or use the offline path. An alert is not a hard spending limit. Use a 90-minute
external reminder and a 2-hour maximum pod lifetime if the provider supports it.

### Before provisioning

1. Finish the offline suite. Prepare prompts, run script, result schema, and the exact
   checkpoint/tokenizer manifest on your own machine. Never upload private prompts.
2. Check model license and weight integrity; use trusted safetensors when possible.
   You supply the model under `/workspace/models/<your-checkpoint>` through your own
   approved process. The lab never fetches weights and does not set up Hub credentials.
3. Verify the image's Python/CUDA/driver/vLLM/torch compatibility **before billing**.
   Keep its digest and package versions. Do not run `uv sync` against this workspace
   just to resolve a vendor image. Use a separate target-host environment.
4. Set a provider budget/alert and record the pod/volume IDs. Do not expose notebook,
   SSH, inference, or metrics ports publicly; restrict access to your account/IP.
   Do not embed credentials in logs, code, local model paths, or JSON reports.

### Execute a bounded experiment

1. Start the selected pod and verify GPU type/memory with `nvidia-smi`. Record driver,
   software versions, image digest, cost/hour, and UTC start time. If wrong, terminate.
2. Copy only this project and approved local model directory. Set `PYTHONPATH` to its
   `src`. Confirm offline env flags and `trust_remote_code=False`; missing files should
   fail locally rather than trigger a download.
3. Run `python -m vllm_benchmark demo`, then one HF request with `--warmup 0 --repeats 1
   --max-new-tokens 8`. Run one vLLM request in a fresh process. Verify token counts and
   `evidence: instrumented_local_model_measurement`. If either adapter is incompatible,
   fix within the remaining budget or stop; do not present demo output as fallback data.
4. Run the README's fixed serial workload, separately for HF and vLLM. Keep cache
   disabled first, then label a second vLLM `--prefix-cache` run as warm-cache. Never
   compare a cold HF run and warm vLLM run without identifying that confound.
5. Monitor `nvidia-smi` and wall time. Store external peak-memory samples if making a
   memory claim (the CLI records GPU capacity, not peak allocation). On OOM, stop the
   process, lower context/output length or model size, and restart. Do not repeatedly
   rent bigger instances without rechecking the cap. GPU utilization 0.8 is a runtime
   allocation setting, **not** a guarantee that the process will fit or share safely.
6. Save compact JSON reports plus manifest and failure notes. Report sample counts,
   serial scope, instrumentation, and actual elapsed cost. End after 2 hours even if
   the intended sweep is incomplete. No cloud execution is needed to pass this week.

### Teardown — mandatory even after an error

- Stop Python generation and any user-started services; check `nvidia-smi` for remaining
  workers. Closing your terminal or stopping a local process **does not stop billing**.
- Copy only sanitized reports to your machine and verify they open. No model weights,
  prompts, API keys, or giant caches need to be retained for this lab report.
- In the RunPod console **terminate/delete the pod**, rather than merely disconnecting
  or stopping it. Separately delete unneeded network volumes, snapshots, and endpoints;
  persistent storage can continue billing after GPU termination.
- Confirm no active pods/endpoints and no unwanted volumes remain in the provider's
  resource inventory. Check the billing dashboard again after its reporting delay;
  record teardown time, final usage, and any continuing storage charge.
- Revoke any temporary credentials used for your own provisioning. An idle/paused GPU
  is not teardown. Set a next-day billing reminder if final costs lag.

## Claims this budget cannot support

No multi-GPU throughput study, public Groq/Cerebras/SambaNova API comparison, production
SLO, GPTQ/AWQ model-quality result, or measured speculative speedup is shipped. Those
need their own licenses, fair workload, provider-specific accounting, and authorization.
The offline probability and allocator tests are still real correctness evidence.
