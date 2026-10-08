# %% [markdown]
# # Track L beta: protocol boundaries, independent verification and token ledgers
# Everything here runs offline. No google-adk import, model call or SWE-Bench data.

# %%
from protocol_lab import (
    A2AClient,
    A2AServer,
    CacheKey,
    EvaluationAttempt,
    LocalTransport,
    MCPClient,
    MCPServer,
    ModelOption,
    Price,
    ResponseCache,
    TokenUsage,
    estimate_cost,
    evaluate,
    repository_fixtures,
    route_model,
)
from protocol_lab.rpc import JSONObject


def text_content(result: JSONObject) -> str:
    content = result.get("content")
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        raise AssertionError("invalid content")
    text = content[0].get("text")
    if not isinstance(text, str):
        raise AssertionError("invalid text")
    return text


wire = LocalTransport(MCPServer())
client = MCPClient(wire)
print("initialize:", client.initialize())
print("tools:", [tool["name"] for tool in client.list_tools()])
result = client.call_tool("add", {"a": 2, "b": 3})
assert text_content(result) == "5"
assert not result["isError"]
print("serialized request:", wire.requests[-1])
print("serialized result:", result)

# %% [markdown]
# MCP is an Anthropic-origin protocol. This server is an explicitly reduced
# 2024-11-05 teaching subset. A2A and ADK are Google Cloud initiatives, not DeepMind.
# The local A2A-shaped demo below is teaching subset v0.1, NOT a compliant wire API.

# %%
a2a = A2AClient(LocalTransport(A2AServer()))
print("submit:", a2a.submit("demo-1", "Summarize the public guide")["status"])
assert a2a.advance("demo-1")["status"]["state"] == "working"
finished = a2a.advance("demo-1")
assert finished["status"]["state"] == "completed"
print("artifact:", finished["artifacts"])

# %%
fixtures = repository_fixtures()
attempts = [
    EvaluationAttempt(
        "config-timeout",
        {"config.json": '{"timeout_seconds": 30}'},
        steps=2,
        calls=1,
        input_tokens=1000,
        output_tokens=100,
    ),
    EvaluationAttempt(
        "readme-version",
        {"README.md": "# Widget\nVersion: 2\n"},
        steps=2,
        calls=1,
        denied_calls=1,
        input_tokens=500,
        output_tokens=50,
    ),
]
report = evaluate(fixtures, attempts)
assert report.task_success == 1
assert report.safe_success == 0.5
print("evaluation:", report)
assert "not SWE-Bench" in report.scope

# %% [markdown]
# Hypothetical prices, not any provider's current rates. Input is partitioned
# into 400 fresh, 400 cached, and 200 cache-write tokens. Writes use an all-in rate.

# %%
usage = TokenUsage(input_tokens=1000, cached_tokens=400, cache_write_tokens=200, output_tokens=100)
price = Price(
    input_per_million=2, cached_per_million=0.5, cache_write_per_million=2.5, output_per_million=8
)
cost = estimate_cost(usage, price)
assert abs(cost - 0.0023) < 1e-12
assert abs(estimate_cost(usage, price, batch_factor=0.5) - 0.00115) < 1e-12
print("hypothetical request cost:", cost)
options = [
    ModelOption("small", Price(1, 0.2, 1.25, 4), quality=0.7, latency_ms=60),
    ModelOption("large", price, quality=0.9, latency_ms=150),
]
chosen = route_model(options, usage, min_quality=0.8, max_latency_ms=200, max_cost=0.01)
assert chosen.name == "large"
print("eligible route:", chosen.name)

# %%
clock = [0.0]
cache = ResponseCache(capacity=2, ttl_seconds=10, clock=lambda: clock[0])
key = CacheKey(
    "authenticated-user-1",
    "pinned-model",
    "public guide",
    "corpus-v1",
    "tools-v1",
    '{"temperature":0}',
)
cache.put(key, {"answer": "Public fixtures only."})
assert cache.get(key) is not None
clock[0] = 10
assert cache.get(key) is None
print("cache expired at the documented boundary")

# %% [markdown]
# Optional Google Cloud ADK sample lives in protocol_lab/adk_sample.py. It uses
# real Agent/InMemoryRunner APIs with lazy imports. It is not invoked here.
# Construct only in a separate environment; model execution requires an explicit
# --run-model opt-in, credentials, quotas, and acceptance of API charges.
