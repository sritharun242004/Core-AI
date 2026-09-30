# Track L beta — protocols, evaluation and economics

A dependency-free Python 3.13 reference for **serialized local MCP interactions**,
a deliberately reduced **A2A-shaped task lifecycle**, an optional genuine **Google
Cloud ADK** sample, local repository-task evaluation, routing and token economics.

**Attribution:** MCP originated at Anthropic. A2A and ADK are Google Cloud
initiatives, not Google DeepMind products. The protocol exercises are explicitly
teaching subsets, not certifications of standards compliance or interoperability.

## Run everything required offline

From the repository root using its existing environment:

```bash
PYTHONPATH=projects/week-23l-mcp-a2a-adk-lab/src .venv/bin/python -m pytest projects/week-23l-mcp-a2a-adk-lab/tests
PYTHONPATH=projects/week-23l-mcp-a2a-adk-lab/src .venv/bin/python projects/week-23l-mcp-a2a-adk-lab/notebooks/01_protocols_and_economics.py
.venv/bin/ruff check projects/week-23l-mcp-a2a-adk-lab
```

No downloads, subprocess tool execution, sockets, credentials or cloud are needed.
Tests use StringIO streams, serialized loopback, malformed fake transports and a
fake optional ADK module. The percent notebook is directly executable Python.

## MCP: a real JSON wire boundary, a limited protocol

`MCPServer` and `MCPClient` communicate through `Transport.exchange(str)`, never by
passing Python request dictionaries directly. `LocalTransport` serializes across
that boundary. `serve_lines(server, source, sink)` also supports newline-delimited
stdio; run the server with:

```bash
PYTHONPATH=projects/week-23l-mcp-a2a-adk-lab/src .venv/bin/python -m protocol_lab
```

Send one JSON object per line (Ctrl-D ends an interactive session):

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"learner","version":"1"}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"add","arguments":{"a":2,"b":3}}}
```

Only requests get responses. The initialized notification must follow a successful
initialize exchange. `tools/list` advertises `add` and `lookup` with exact-key
object schemas; `tools/call` validates finite numbers within ±1,000 and strings
at most 128 characters. Tools only read local public fixtures or add numbers.
The host caps attempted tool calls at 64/session by default, including invalid
attempts. An unrecognized tool is not a way to install a new capability.

| Contract | Behavior |
| --- | --- |
| Protocol reference | MCP **2024-11-05**, teaching subset only |
| Malformed JSON / oversized frame | JSON-RPC `-32700` |
| Invalid request envelope / repeated initialization | `-32600` |
| Unknown method | `-32601` |
| Invalid schema / tool name / unsupported version | `-32602` |
| Before initialization is complete | local `-32002` |
| Tool budget exhausted | local `-32000` |
| Public lookup missing | successful RPC with tool result `isError: true` |
| Tool result | `content` list containing bounded text blocks; boolean `isError` |

Frames are capped at 65,536 characters. JSON nonfinite constants, mismatched IDs,
both-result-and-error responses and malformed content are rejected. No JSON-RPC
batch arrays, MCP resources/prompts, sampling, subscriptions, cancellation,
pagination, auth, protocol negotiation fallback or HTTP transport are implemented.
The schema validator supports only the schemas actually advertised, **not full
JSON Schema**. Compare against the pinned specification before adding a real SDK.
No claim that an arbitrary MCP client accepts all these reduced behaviors.

## A2A: honest task lifecycle teaching subset v0.1

`A2AServer`/`A2AClient` use the same serialized transport. Their **local** method
surface is `agent/card`, `tasks/send`, `tasks/get`, `tasks/cancel`, and `demo/advance`.
The last is a deterministic test scheduler, not an A2A standard method. Method
names/envelopes are not promised to match any released A2A revision.

```
submitted → working → completed | failed
    └──────────┴────→ canceled
```

Terminal states cannot mutate. Same ID + same request is idempotent; same ID +
different request is a conflict. Tasks and text are bounded. The worker can only
return a public guide fixture; unsupported tasks fail. No network discovery,
Agent Card standard conformance, authentication, streaming, push notifications,
remote delegation or actual model reasoning is claimed. One server instance is
one trusted local session; production needs authenticated principal isolation.

## Optional genuine Google Cloud ADK sample

`protocol_lab/adk_sample.py` imports `google.adk.agents.Agent` lazily and constructs
an actual ADK agent with a typed public-record function tool. The optional model
path uses `InMemoryRunner`, its session service, `google.genai.types.Content`, and
`RunConfig(max_llm_calls=2)`. This is real API wiring, not a homegrown class named
ADK. The offline test replaces the imported module to check the builder's contract;
**the third-party package and cloud path were not installed/executed for validation**.

For optional exploration only, use a **separate disposable environment** and install
this project's `[adk]` extra (`google-adk==1.18.0`), not the shared workspace:

```bash
# In that isolated environment, after installing the optional extra:
python -m protocol_lab.adk_sample --model YOUR_AVAILABLE_GEMINI_MODEL
# Above constructs an Agent only; it does not invoke the model.
# Explicit paid opt-in after credential and quota setup:
python -m protocol_lab.adk_sample --model YOUR_AVAILABLE_GEMINI_MODEL --run-model
```

The default model string is illustrative and may retire; explicitly select a
currently available model. Follow the pinned ADK docs for Gemini API or Vertex AI
credential configuration. Keep secrets out of source/notebooks. The sample limits
model calls and uses a best-effort 30-second coroutine deadline; provider quotas
and a separate spend budget remain necessary. See `COMPUTE.md` before any opt-in.

## Repository evaluation — not SWE-Bench

`repository_fixtures()` returns two tiny repositories as in-memory file maps:
repair a JSON timeout and synchronize a README version with a package manifest.
`EvaluationAttempt` carries allowlisted whole-file replacements and trace counters.
`evaluate` applies replacements in memory, independently checks the resulting
files, and reports per-case reasons, task/safe success, denied-attempt rate,
steps/calls and input/output tokens. It never runs submitted code or shell commands.

Missing attempts stay in the denominator; duplicate/unknown IDs are errors;
unsafe paths, non-editable files and budget overruns fail. The trace counters must
come from trusted host logs in a real system, not model self-report. These tiny
fixtures verify harness behavior, **not an actual SWE-Bench Verified score**, a
model ranking, a coding-agent benchmark, or an Inspect AI integration. Real
SWE-Bench requires official tasks/versions, sandboxed repository environments and
an independently pinned test harness; real Inspect AI uses its own task/solver/
scorer APIs. Neither is impersonated here.

## Cost, routing and caching API

- `TokenUsage`: disjoint fresh, cached-read and cache-write input partitions plus
  output tokens. `Price`: hypothetical dollars per million, not live provider prices.
- `estimate_cost`: all-in write price and an optional illustrative batch multiplier.
  Real providers may combine discounts differently; never infer a current bill.
- `ModelOption`/`route_model`: cheapest eligible model under explicit validation
  quality, latency and cost thresholds. No eligible model raises rather than
  silently weakening constraints. Quality/latency are supplied measurements or
  fixtures, not predictions of an unknown future request.
- `CacheKey`: authenticated tenant/principal scope, model, prompt, context revision,
  tool revision and generation parameters. `ResponseCache`: bounded TTL/LRU,
  defensive copies, JSON-size limits and no failed-response insertion.
  This is exact **response caching**, not a provider KV/prefix-cache implementation.

Learn conceptual framework tradeoffs (LangGraph, CrewAI, AutoGen, OpenAI Agents SDK)
and knowledge-graph/context engineering in the accompanying lesson. This package
does not create counterfeit framework implementations.

Primary sources: [MCP 2024-11-05](https://modelcontextprotocol.io/specification/2024-11-05),
[A2A](https://a2a-protocol.org/latest/), [ADK](https://google.github.io/adk-docs/),
[SWE-Bench](https://www.swebench.com/), [Inspect AI](https://inspect.aisi.org.uk/).
