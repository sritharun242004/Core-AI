# Week 19 — inference engines without benchmark fiction

A Python **3.13**, NumPy-only correctness lab plus opt-in, local-checkpoint HF/vLLM
measurement CLI. Prerequisite: `week-18-fsdp-ring-lab`. The full inference-engine
comparison is **red tier**; the required offline path costs **$0**, takes seconds,
and needs no GPU, credentials, network, model weights, or optional framework imports.

## Run the offline path

From the repository root, using the existing environment (do not resync the workspace):

```bash
ROOT="$PWD"
cd projects/week-19-vllm-benchmark
export PYTHONPATH="$PWD/src"
"$ROOT/.venv/bin/python" -m pytest
"$ROOT/.venv/bin/ruff" check .
"$ROOT/.venv/bin/python" notebooks/01-inference-contracts.py
"$ROOT/.venv/bin/python" -m vllm_benchmark demo
```

`demo --output /tmp/week19-simulation.json` saves a report with
`evidence: simulation_not_hardware_measurement`. Its timestamps are invented numeric
fixtures. Its 4 output tokens/second is **not an engine benchmark**. No clock around
NumPy arithmetic is advertised as model latency.

## Public API and boundaries

| API | Contract | What it does not implement |
|---|---|---|
| `quantize(x, bits=4 or 8, group_size=64)` / `dequantize(q)` | Symmetric last-axis groups, partial tail groups, float32 scales, ties-to-even rounding; zero group uses scale 1 | GPTQ, AWQ, calibration, packed INT4 kernels, model-quality evaluation |
| `QuantizedWeights.estimated_packed_bytes` | Ideal bit payload + 4 bytes/scale | Actual Python/array allocation, alignment, kernel speedup; INT4 codes occupy int8 elements |
| `PagedKVCache(num_pages, page_size)` | Atomic allocate/append, release, immutable page-table views, no page aliasing, free+allocated=capacity | GPU memory, shared pages, refcounts, eviction, copy-on-write, attention kernels |
| `CacheIdentity` / `prefix_key(identity, ids)` | SHA-256 of canonical versioned JSON containing tenant, weight/tokenizer/adapter revisions, position config, KV dtype, exact token sequence | Authentication, encryption, semantic similarity, multimodal identity, automatic longest-prefix search |
| `PrefixCache(max_entries)` | Exact-entry, tenant-scoped LRU; `None` means miss; `clear()` drops all entries | Byte budget, TTL, secure deletion, distributed consistency |
| `RequestTrace` / `aggregate_metrics` | Validated seconds, explicit complete wall window, absent TTFT/TPOT as `None` | Failed-request success metrics, steady-state load testing, kernel profiling |
| `acceptance_probability`, `residual_distribution`, `speculative_sample` | Exact one-step target/draft sampling, explicit seeded RNG | Multi-token model execution, speculative speedup, tokenizer remapping |

For text-only KV reuse, the caller's `position_config` must include every nondefault
position/mask policy. Include new inputs (images, audio, token type IDs, model-specific
conditioning) in the identity before extending beyond this text-only fixture. A hash
is neither an authorization boundary nor privacy protection against guessing.

## Metric definitions

For submitted time `s`, output observation times `t[0] ... t[n-1]`, and completion `c`:

- TTFT = `t[0] - s` if `n >= 1`, otherwise absent.
- TPOT = `(t[n-1] - t[0]) / (n-1)` if `n >= 2`, otherwise absent.
- End-to-end request latency = `c - s`, including completion overhead.
- Output throughput = **all output tokens, including first tokens** / explicit wall
  window. Do not sum request durations when requests overlap. Input tokens are excluded.
- Request throughput = completed requests / wall window; empty successful completions
  still count. Failures abort this CLI; no partial successful report is emitted.
- Weighted TPOT sums decode spans and divides by total decode intervals. Macro request
  TPOT weights requests equally; both are reported and can disagree.
- Percentiles use NumPy's default linear interpolation. Tiny-sample p95 is descriptive,
  not a stable tail-latency claim. Never silently exclude timeouts in a production report.

## Optional measured comparison: explicit local weights only

**Not run as part of this offline lab.** Review [COMPUTE.md](COMPUTE.md) before renting
anything. Provision dependencies in a separate compatible environment, not by changing
this repository's lockfile. HF uses optional `torch`/`transformers`; vLLM requires a
compatible Linux/CUDA/vLLM/Python combination. Python 3.13 support and wheels are
release-specific: verify before paying. There is no default checkpoint, model-ID
argument, automatic download, or `trust_remote_code=True` path. Use vetted local
weights/tokenizer/config, ideally safetensors, with license and integrity checks.

Create `prompts.json` containing a JSON list of non-sensitive strings. Start with
ordinary decoder-only text models that fit your device. The CLI uses
`add_special_tokens=True` and **does not apply a chat template**: serialize a versioned
chat template yourself if needed, avoid doubled BOS tokens, and compare input-token
hashes. Set `MODEL_DIR` to an already provisioned local directory and `REVISION` to its
immutable manifest/checkpoint identifier. For each engine, in a **fresh process**:

```bash
python -m vllm_benchmark benchmark --engine hf \
  --model-path "$MODEL_DIR" --revision-label "$REVISION" \
  --prompts-file prompts.json --device cuda --max-model-len 1024 \
  --max-new-tokens 32 --warmup 2 --repeats 10 --output /tmp/week19-hf.json

python -m vllm_benchmark benchmark --engine vllm \
  --model-path "$MODEL_DIR" --revision-label "$REVISION" \
  --prompts-file prompts.json --device cuda --max-model-len 1024 \
  --max-new-tokens 32 --warmup 2 --repeats 10 --output /tmp/week19-vllm.json
```

Use the selected environment's Python and keep `PYTHONPATH=project/src`. HF can use
`--device cpu` for a smoke test (float32), but **do not compare CPU HF against GPU vLLM
as an engine-only claim**. CUDA paths request float16. vLLM uses eager execution,
TP=1, and 80% GPU-memory utilization by default. Record versions and change one knob
at a time. The adapter targets the public `EngineArgs`/`LLMEngine` step interface;
version incompatibility should fail visibly, not substitute synthetic results.

### What the measured report means

- Concurrency is **1**, prompts run serially, greedy decoding, no multi-request batching
  or arrival-process generator. vLLM's continuous-batching advantage is not tested.
- Loading, tokenization, and warmup are excluded from the measured wall window.
  Request time includes input-tensor submission and engine/generation overhead.
- HF records token-ID streamer callbacks after device synchronization, skipping the
  prompt callback. This overhead can perturb latency; it does not confuse text chunks
  with tokens. vLLM timestamps cumulative-token deltas after `step()` returns. Several
  tokens in a step share an observation time. Neither is uninstrumented kernel time.
- Generation can stop at EOS before `max_new_tokens`; compare actual token counts,
  EOS configuration, tokenizer hash, device, dtype, and model revision, not just limits.
  Greedy outputs can still differ across kernels; this is not a quality comparison.
- Prefix caching is off by default. `--prefix-cache` is vLLM-only and makes repeated
  prompts/warmup a **warm-cache** workload. For cold-cache comparisons disable it;
  enabling it with `warmup=0` does not keep repeated requests cold.
- Reports contain software versions, checkpoint label, local model path, hardware,
  token counts, relative token timings, and input-token hash, not raw prompts or text.
  Local paths and hashes can still be sensitive: scrub before publishing.

For a genuine serving comparison, add a concurrency/load sweep (1/4/16/64), fixed
arrival-rate and length distributions, queue time, goodput under TTFT/TPOT SLOs,
failures, GPU-memory samples, warm/cold cache controls, and quality checks. Do not
publish a vendor leaderboard from this serial harness. See the three
[assignments](assignments/) and [worked notes](SOLUTION_NOTES.md).
