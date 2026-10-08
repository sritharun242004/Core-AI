import pytest
from protocol_lab import (
    CacheKey,
    EvaluationAttempt,
    ModelOption,
    Price,
    ResponseCache,
    TokenUsage,
    estimate_cost,
    evaluate,
    repository_fixtures,
    route_model,
)


def test_repository_fixtures_check_actual_patches_and_trajectory():
    fixtures = repository_fixtures()
    attempts = [
        EvaluationAttempt(
            "config-timeout", {"config.json": '{"timeout_seconds": 30}'}, steps=2, calls=1
        ),
        EvaluationAttempt(
            "readme-version",
            {"README.md": "# Widget\nVersion: 2\n"},
            steps=2,
            calls=1,
            denied_calls=1,
        ),
    ]
    report = evaluate(fixtures, attempts)
    assert report.task_success == 1.0
    assert report.safe_success == 0.5
    assert report.mean_steps == 2
    assert report.total_calls == 2
    assert report.denied_attempt_rate == 0.5
    assert report.suite == "local-repository-fixtures-v1"
    assert "not SWE-Bench" in report.scope


def test_wrong_patch_and_path_escape_are_failures_not_executed_code():
    fixtures = repository_fixtures()
    report = evaluate(
        fixtures,
        [
            EvaluationAttempt("config-timeout", {"../secret": "x"}),
            EvaluationAttempt("readme-version", {"README.md": "wrong"}),
        ],
    )
    assert report.task_success == 0
    assert report.results[0].reason == "invalid_patch"
    assert report.results[1].reason == "verification_failed"


def test_missing_attempt_is_in_denominator_and_empty_suite_rejected():
    report = evaluate(repository_fixtures(), [])
    assert report.task_success == 0
    assert len(report.results) == 2
    with pytest.raises(ValueError):
        evaluate([], [])


def test_budget_excess_and_duplicates_cannot_inflate_score():
    fixtures = repository_fixtures()
    attempt = EvaluationAttempt(
        "config-timeout", {"config.json": '{"timeout_seconds": 30}'}, steps=99
    )
    report = evaluate(fixtures, [attempt])
    assert report.results[0].reason == "budget_exceeded"
    with pytest.raises(ValueError):
        evaluate(fixtures, [attempt, attempt])
    with pytest.raises(ValueError):
        evaluate(fixtures, [EvaluationAttempt("unknown", {})])


def test_fixture_verifier_ignores_claimed_success_and_rejects_wrong_types():
    attempt = EvaluationAttempt("config-timeout", {"config.json": '{"timeout_seconds": true}'})
    assert evaluate(repository_fixtures(), [attempt]).task_success == 0


def test_cost_distinguishes_fresh_cached_write_and_output_tokens():
    usage = TokenUsage(
        input_tokens=1000, cached_tokens=400, cache_write_tokens=200, output_tokens=100
    )
    price = Price(
        input_per_million=2,
        cached_per_million=0.5,
        cache_write_per_million=2.5,
        output_per_million=8,
    )
    # 400 fresh*2 + 400 cached*.5 + 200 writes*2.5 + 100 output*8 = 2300 / 1e6
    assert estimate_cost(usage, price) == pytest.approx(0.0023)
    assert estimate_cost(usage, price, batch_factor=0.5) == pytest.approx(0.00115)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_tokens": -1},
        {"input_tokens": True},
        {"input_tokens": 3, "cached_tokens": 4},
        {"input_tokens": 3, "cached_tokens": 2, "cache_write_tokens": 2},
    ],
)
def test_invalid_token_ledgers(kwargs: dict[str, int | bool]) -> None:
    with pytest.raises(ValueError):
        TokenUsage(**kwargs)


def test_invalid_prices_and_batch_factors():
    with pytest.raises(ValueError):
        Price(float("nan"), 0, 0, 0)
    with pytest.raises(ValueError):
        estimate_cost(TokenUsage(), Price(1, 1, 1, 1), batch_factor=0)


def test_router_selects_cheapest_eligible_model_or_explicitly_fails():
    options = [
        ModelOption("small", Price(1, 1, 1, 1), quality=0.7, latency_ms=50),
        ModelOption("large", Price(5, 5, 5, 5), quality=0.9, latency_ms=150),
    ]
    usage = TokenUsage(input_tokens=1000, output_tokens=100)
    assert (
        route_model(options, usage, min_quality=0.8, max_latency_ms=200, max_cost=1).name == "large"
    )
    assert (
        route_model(options, usage, min_quality=0.6, max_latency_ms=100, max_cost=1).name == "small"
    )
    with pytest.raises(ValueError, match="no eligible"):
        route_model(options, usage, min_quality=0.95, max_latency_ms=200, max_cost=1)


def key(tenant: str = "one", revision: str = "r1") -> CacheKey:
    return CacheKey(tenant, "model-v1", "prompt", revision, "tools-v1", '{"temperature":0}')


def test_cache_is_partitioned_versioned_expiring_and_defensively_copied():
    clock = [0.0]
    cache = ResponseCache(capacity=2, ttl_seconds=10, clock=lambda: clock[0])
    response = {"answer": ["public"]}
    cache.put(key(), response)
    response["answer"].append("mutated")
    assert cache.get(key()) == {"answer": ["public"]}
    assert cache.get(key("two")) is None
    assert cache.get(key(revision="r2")) is None
    found = cache.get(key())
    assert found is not None
    answer = found.get("answer")
    assert isinstance(answer, list)
    answer.append("mutated again")
    assert cache.get(key()) == {"answer": ["public"]}
    clock[0] = 10
    assert cache.get(key()) is None


def test_cache_lru_and_failed_responses():
    cache = ResponseCache(capacity=1)
    cache.put(key(), {"answer": "one"})
    cache.put(key("two"), {"answer": "two"}, success=False)
    assert cache.get(key()) == {"answer": "one"}
    cache.put(key("two"), {"answer": "two"})
    assert cache.get(key()) is None
    assert cache.get(key("two")) == {"answer": "two"}


def test_cache_rejects_non_json_and_oversize_values():
    cache = ResponseCache(max_bytes=30)
    with pytest.raises(ValueError):
        cache.put(key(), {"answer": "x" * 31})
    with pytest.raises((ValueError, TypeError)):
        cache.put(key(), {"answer": float("nan")})
