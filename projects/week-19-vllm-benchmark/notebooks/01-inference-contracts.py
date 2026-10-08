# %% [markdown]
# # Week 19 — four inference contracts, zero rented GPUs
# Run with PYTHONPATH=src python notebooks/01-inference-contracts.py from the project.
# This percent-format notebook can be converted with jupytext if installed.
# Every latency below is a numeric fixture, not a hardware/model measurement.

# %%
import json

import numpy as np
from vllm_benchmark import (
    CacheIdentity,
    PagedKVCache,
    PrefixCache,
    RequestTrace,
    acceptance_probability,
    aggregate_metrics,
    dequantize,
    quantize,
    residual_distribution,
    speculative_sample,
)

print("SIMULATION: no inference engine, checkpoint, or hardware timing is used.")

# %% [markdown]
# ## 1. Quantization: error and storage are different claims
# Grouping isolates outliers. INT4 codes here are unpacked int8, not a GPU format.

# %%
weights = np.array([[-1.0, 0.0, 0.5, 1.0, 70.0]], dtype=np.float32)
for bits in (8, 4):
    result = quantize(weights, bits=bits, group_size=2)
    print(f"INT{bits}", result.codes, "scales", result.scales)
    print("max error:", np.max(np.abs(dequantize(result) - weights)))
    print("ideal packed bytes + scales:", result.estimated_packed_bytes)

# %% [markdown]
# ## 2. KV pages: fail before mutating, reuse only after release
# Three pages hold 12 token slots. A partial last page is internal fragmentation.

# %%
cache = PagedKVCache(num_pages=3, page_size=4)
cache.allocate("A", 5)
cache.allocate("B", 4)
print("tail waste:", cache.internal_waste_tokens)
try:
    cache.append("A", 4)
except MemoryError:
    print("Expected capacity rejection; A still has", cache.tokens_for("A"), "tokens")
cache.append("A", 3)
cache.release("B")
cache.append("A", 1)
cache.check_invariants()
cache.release("A")
assert cache.free_pages == 3

# %% [markdown]
# ## 3. Prefix cache: exact computation identity, not similar text
# Tenant is an authenticated namespace supplied by the caller in a real service.

# %%
identity = CacheIdentity("tenant-a", "weights-sha", "tokenizer-sha", "none", "rope-v1", "fp16")
prefixes = PrefixCache(max_entries=2)
prefixes.put(identity, [1, 23], "opaque-kv-handle")
assert prefixes.get(identity, [1, 23]) == "opaque-kv-handle"
assert prefixes.get(identity, [12, 3]) is None
assert prefixes.get(identity, [1, 23, 4]) is None
print("Exact-entry hits require the same namespace AND entire supplied prefix.")

# %% [markdown]
# ## 4. Metrics: first tokens count in throughput, not decode intervals
# This example uses an explicit [0,1] wall window, including idle time.

# %%
traces = [RequestTrace(0, (0.2, 0.3, 0.4), 0.5), RequestTrace(0.1, (0.6,), 0.8)]
metrics = aggregate_metrics(traces, window_start_s=0, window_end_s=1)
assert metrics["output_tokens_per_s"] == 4
weighted_tpot = metrics["weighted_tpot_s"]
assert weighted_tpot is not None and np.isclose(weighted_tpot, 0.1)
print(json.dumps({"evidence": "simulation_not_hardware_measurement", "metrics": metrics}, indent=2))

# %% [markdown]
# ## 5. Speculation: exact target law, not acceptance-rate theater
# Accepted mass plus rejection mass times residual must reconstruct the target.
# The one-step sampler does not measure any inference acceleration.

# %%
target = np.array([0.6, 0.3, 0.1])
draft = np.array([0.2, 0.3, 0.5])
accept = acceptance_probability(target, draft)
residual = residual_distribution(target, draft)
np.testing.assert_allclose(draft * accept + (1 - np.sum(draft * accept)) * residual, target)
rng = np.random.default_rng(19)
draws = [speculative_sample(target, draft, rng) for _ in range(5000)]
print("accept probabilities:", accept, "residual:", residual)
print("sample frequencies:", np.bincount([draw.token for draw in draws], minlength=3) / len(draws))
print("observed acceptance:", np.mean([draw.accepted for draw in draws]))

# %% [markdown]
# ## Next: real measurements are an explicit opt-in
# Read README.md and COMPUTE.md. The optional CLI accepts only a supplied local model
# directory, never a default Hub model. Compare workload/precision/cache state before
# latency, and tear down rented resources even when a run fails.
