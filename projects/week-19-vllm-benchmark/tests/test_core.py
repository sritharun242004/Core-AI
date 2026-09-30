"""Correctness contracts written before implementation; no models/network required."""

import random

import numpy as np
import pytest
from vllm_benchmark import (
    CacheIdentity,
    PagedKVCache,
    PrefixCache,
    RequestTrace,
    acceptance_probability,
    aggregate_metrics,
    dequantize,
    prefix_key,
    quantize,
    residual_distribution,
    speculative_sample,
)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [1, 3, 8])
def test_group_quantization_error_bound_and_shape(bits, group_size):
    weights = np.array([[-2.1, 0.0, 0.3, 0.7, 8.2], [0.0, 0.0, 0.0, -0.2, 0.1]])
    original = weights.copy()
    packed = quantize(weights, bits=bits, group_size=group_size)
    restored = dequantize(packed)
    assert restored.shape == weights.shape
    assert packed.scales.shape == (2, (5 + group_size - 1) // group_size)
    assert packed.codes.dtype == np.int8  # INT4 values are NOT packed into nibbles.
    assert np.max(np.abs(packed.codes)) <= (2 ** (bits - 1) - 1)
    for start in range(0, 5, group_size):
        error = np.abs(
            restored[:, start : start + group_size] - weights[:, start : start + group_size]
        )
        bound = packed.scales[:, start // group_size, None] / 2
        assert np.all(error <= bound + 1e-6)
    np.testing.assert_array_equal(weights, original)
    expected_bytes = (weights.size * bits + 7) // 8 + packed.scales.size * 4
    assert packed.estimated_packed_bytes == expected_bytes


def test_zero_groups_and_outliers_are_independent():
    packed = quantize(np.array([0.0, 0.0, 0.1, 0.2, 70.0]), bits=4, group_size=2)
    assert packed.scales[0] == 1.0
    assert packed.scales[1] == pytest.approx(0.2 / 7)
    assert packed.scales[2] == 10.0
    np.testing.assert_array_equal(dequantize(packed)[:2], [0, 0])


@pytest.mark.parametrize(
    "values,bits,group",
    [
        ([], 4, 2),
        ([float("nan")], 4, 2),
        ([1.0], 3, 2),
        ([1.0], 4, 0),
        ([1.0], 4, 1.5),
        ([float("inf")], 8, 2),
    ],
)
def test_invalid_quantization(values, bits, group):
    with pytest.raises(ValueError):
        quantize(np.array(values), bits=bits, group_size=group)


def test_paging_fragmentation_reuse_and_atomic_exhaustion():
    cache = PagedKVCache(num_pages=3, page_size=4)
    assert cache.allocate("a", 5) == (0, 1)
    assert cache.allocate("b", 4) == (2,)
    assert cache.free_pages == 0
    assert cache.internal_waste_tokens == 3
    before = cache.pages_for("a")
    with pytest.raises(MemoryError):
        cache.append("a", 4)
    assert cache.pages_for("a") == before
    assert cache.tokens_for("a") == 5
    cache.append("a", 3)  # Fits the already allocated tail page.
    cache.release("b")
    assert cache.append("a", 1) == (0, 1, 2)
    assert cache.internal_waste_tokens == 3
    cache.release("a")
    assert cache.free_pages == 3
    assert cache.internal_waste_tokens == 0
    cache.check_invariants()


def test_randomized_allocator_capacity_and_no_aliasing():
    rng = random.Random(19)
    cache = PagedKVCache(num_pages=9, page_size=3)
    live = {}
    for step in range(300):
        if live and rng.random() < 0.45:
            name = rng.choice(list(live))
            cache.release(name)
            del live[name]
        else:
            count = rng.randrange(0, 15)
            name = f"r{step}"
            if (count + 2) // 3 <= cache.free_pages:
                live[name] = cache.allocate(name, count)
            else:
                with pytest.raises(MemoryError):
                    cache.allocate(name, count)
        pages = [p for allocation in live.values() for p in allocation]
        assert len(pages) == len(set(pages))
        assert len(pages) + cache.free_pages == 9
        cache.check_invariants()


@pytest.mark.parametrize("pages,size", [(0, 4), (4, 0), (-1, 2), (2.5, 4), (True, 4)])
def test_invalid_allocator_dimensions(pages, size):
    with pytest.raises(ValueError):
        PagedKVCache(pages, size)


def test_allocator_lifecycle_errors_do_not_mutate():
    cache = PagedKVCache(2, 4)
    cache.allocate("empty", 0)
    with pytest.raises(ValueError):
        cache.allocate("empty", 1)
    with pytest.raises(ValueError):
        cache.append("empty", -1)
    with pytest.raises(KeyError):
        cache.release("missing")
    assert cache.free_pages == 2
    cache.release("empty")


def identity(**updates):
    fields = dict(
        tenant="tenant-a",
        model_revision="weights-sha",
        tokenizer_revision="tok-sha",
        adapter_revision="none",
        position_config="rope-default",
        kv_dtype="fp16",
    )
    fields.update(updates)
    return CacheIdentity(**fields)


def test_exact_prefix_keys_include_all_semantic_and_security_inputs():
    baseline = prefix_key(identity(), [1, 23])
    assert baseline == prefix_key(identity(), (1, 23))
    assert baseline != prefix_key(identity(), [12, 3])
    assert baseline != prefix_key(identity(), [1, 23, 0])
    assert baseline != prefix_key(identity(), [23, 1])
    for field in (
        "tenant",
        "model_revision",
        "tokenizer_revision",
        "adapter_revision",
        "position_config",
        "kv_dtype",
    ):
        assert baseline != prefix_key(identity(**{field: "changed"}), [1, 23])
    with pytest.raises(ValueError):
        prefix_key(identity(), [1.5])
    with pytest.raises(ValueError):
        prefix_key(identity(tenant=""), [1])


def test_prefix_cache_is_exact_tenant_scoped_and_bounded_lru():
    cache = PrefixCache(max_entries=2)
    cache.put(identity(), [1, 2], "kv-a")
    cache.put(identity(), [3], "kv-b")
    assert cache.get(identity(), [1, 2]) == "kv-a"
    assert cache.get(identity(), [1, 2, 3]) is None
    assert cache.get(identity(tenant="other"), [1, 2]) is None
    cache.put(identity(), [4], "kv-c")
    assert cache.get(identity(), [3]) is None
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0


def test_timing_denominators_use_output_tokens_and_explicit_wall_window():
    a = RequestTrace(0.0, (0.2, 0.3, 0.4), 0.5)
    b = RequestTrace(0.1, (0.6,), 0.8)
    empty = RequestTrace(0.3, (), 0.9)
    assert a.ttft_s == pytest.approx(0.2)
    assert a.tpot_s == pytest.approx(0.1)  # (0.4 - 0.2) / (3 - 1), NOT / 3.
    assert b.tpot_s is None
    assert empty.ttft_s is None
    report = aggregate_metrics([a, b, empty], window_start_s=0, window_end_s=1.0)
    assert report["output_tokens"] == 4
    assert report["output_tokens_per_s"] == 4.0
    assert report["completed_requests_per_s"] == 3.0
    assert report["mean_ttft_s"] == pytest.approx(0.35)
    assert report["weighted_tpot_s"] == pytest.approx(0.1)
    assert report["ttft_samples"] == 2
    assert report["decode_intervals"] == 2


def test_aggregate_tpot_weights_decode_intervals_not_requests():
    traces = [RequestTrace(0, (1, 2), 2), RequestTrace(0, (1, 1.1, 1.2, 1.3), 1.3)]
    report = aggregate_metrics(traces, window_start_s=0, window_end_s=2)
    assert report["weighted_tpot_s"] == pytest.approx(1.3 / 4)
    assert report["mean_request_tpot_s"] == pytest.approx(0.55)
    assert report["output_tokens_per_s"] == 3.0
    assert aggregate_metrics([], window_start_s=0, window_end_s=1)["mean_ttft_s"] is None


@pytest.mark.parametrize(
    "submitted,times,completed",
    [(1, (0.5,), 2), (0, (1, 0.5), 2), (0, (1,), 0.9), (0, (), -1), (0, (float("nan"),), 2)],
)
def test_invalid_traces(submitted, times, completed):
    with pytest.raises(ValueError):
        RequestTrace(submitted, times, completed)


def test_invalid_metric_window():
    with pytest.raises(ValueError):
        aggregate_metrics([], window_start_s=1, window_end_s=1)
    with pytest.raises(ValueError):
        aggregate_metrics([RequestTrace(0, (1,), 1)], window_start_s=0.5, window_end_s=2)


def test_speculative_residual_and_exact_distribution_identity():
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.2, 0.3, 0.5])
    accept = acceptance_probability(p, q)
    np.testing.assert_allclose(accept, [1, 1, 0.2])
    residual = residual_distribution(p, q)
    np.testing.assert_allclose(residual, [1, 0, 0], atol=1e-15)
    rejection_mass = 1 - np.sum(q * accept)
    np.testing.assert_allclose(q * accept + rejection_mass * residual, p)


def test_speculative_zero_draft_support_and_identical_distributions():
    p, q = np.array([0.0, 0.5, 0.5]), np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(residual_distribution(p, q), [0, 0.5, 0.5])
    np.testing.assert_allclose(q * acceptance_probability(p, q), [0, 0, 0])
    np.testing.assert_array_equal(residual_distribution(p, p), p)  # Unreachable fallback.
    result = speculative_sample(p, p, np.random.default_rng(19))
    assert result.accepted


def test_speculative_sampling_matches_target_not_draft():
    rng = np.random.default_rng(19)
    p, q = np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.2, 0.7])
    samples = [speculative_sample(p, q, rng) for _ in range(12000)]
    counts = np.bincount([s.token for s in samples], minlength=3) / len(samples)
    np.testing.assert_allclose(counts, p, atol=0.015)
    assert np.mean([s.accepted for s in samples]) == pytest.approx(0.4, abs=0.02)


@pytest.mark.parametrize(
    "p,q",
    [
        ([0.5], [0.2, 0.8]),
        ([0.2, 0.2], [0.5, 0.5]),
        ([-0.1, 1.1], [0.5, 0.5]),
        ([float("nan"), 0], [1, 0]),
    ],
)
def test_invalid_speculative_distributions(p, q):
    with pytest.raises(ValueError):
        residual_distribution(p, q)
