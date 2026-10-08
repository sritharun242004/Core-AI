"""Transport and protocol boundary tests; no subprocesses, sockets, or credentials."""

import io
import json
import sys
import types
from collections.abc import Callable
from typing import cast

import pytest
from protocol_lab import (
    A2AClient,
    A2AServer,
    LocalTransport,
    MCPClient,
    MCPServer,
    ProtocolError,
    serve_lines,
)
from protocol_lab.adk_sample import build_agent
from protocol_lab.rpc import JSONObject, decode


def request(method: str, params: JSONObject | None = None, ident: int | str = 1) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": ident, "method": method, "params": params or {}})


def response(wire: str | None) -> JSONObject:
    if wire is None:
        raise AssertionError("expected response")
    value = decode(wire)
    if not isinstance(value, dict):
        raise AssertionError("expected JSON object response")
    return value


def error_code(value: JSONObject) -> int:
    error = value.get("error")
    if not isinstance(error, dict) or type(error.get("code")) is not int:
        raise AssertionError("expected error response")
    return cast(int, error["code"])


def text_content(value: JSONObject) -> str:
    content = value.get("content")
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        raise AssertionError("expected text content")
    text = content[0].get("text")
    if not isinstance(text, str):
        raise AssertionError("expected text content")
    return text


def mcp_pair() -> tuple[MCPClient, LocalTransport]:
    transport = LocalTransport(MCPServer())
    client = MCPClient(transport)
    client.initialize()
    return client, transport


def test_mcp_initialize_list_and_call_cross_json_transport():
    client, transport = mcp_pair()
    tools = client.list_tools()
    assert [tool["name"] for tool in tools] == ["add", "lookup"]
    schema = tools[0].get("inputSchema")
    assert isinstance(schema, dict)
    assert schema.get("additionalProperties") is False
    result = client.call_tool("add", {"a": 2, "b": 3})
    assert result == {"content": [{"type": "text", "text": "5"}], "isError": False}
    assert len(transport.requests) == 4  # initialize, initialized notification, list, call
    assert all(isinstance(item, str) for item in transport.requests)


@pytest.mark.parametrize(
    "wire,code",
    [
        ("{bad", -32700),
        ("[]", -32600),
        ("null", -32600),
        (json.dumps({"jsonrpc": "1.0", "id": 1, "method": "x"}), -32600),
        (request("tools/list"), -32002),
    ],
)
def test_mcp_bad_frames_and_preinitialization(wire: str, code: int) -> None:
    result = response(MCPServer().handle(wire))
    assert error_code(result) == code


def test_mcp_wrong_version_and_unknown_method():
    server = MCPServer()
    response_body = response(
        server.handle(request("initialize", {"protocolVersion": "2099-01-01"}))
    )
    assert error_code(response_body) == -32602
    client, _ = mcp_pair()
    with pytest.raises(ProtocolError) as error:
        client.request("does/not/exist")
    assert error.value.code == -32601


@pytest.mark.parametrize(
    "args",
    [
        {"a": True, "b": 1},
        {"a": 1},
        {"a": 1, "b": 2, "c": 3},
        {"a": float("inf"), "b": 2},
        {"a": 1001, "b": 1},
    ],
)
def test_mcp_schema_errors_are_invalid_params(args: JSONObject) -> None:
    client, _ = mcp_pair()
    with pytest.raises((ProtocolError, ValueError)):
        client.call_tool("add", args)


def test_tool_execution_error_is_not_jsonrpc_error():
    client, _ = mcp_pair()
    result = client.call_tool("lookup", {"key": "private"})
    assert result["isError"] is True
    assert text_content(result) == "public record not found"
    with pytest.raises(ProtocolError) as error:
        client.call_tool("shell", {"command": "not executed"})
    assert error.value.code == -32602


def test_stdio_framing_accepts_notification_without_reply():
    frames = [
        request("initialize", {"protocolVersion": "2024-11-05"}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        request("tools/list", ident=2),
    ]
    output = io.StringIO()
    serve_lines(MCPServer(), io.StringIO("\n".join(frames) + "\n"), output)
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [reply["id"] for reply in replies] == [1, 2]


def test_unknown_notifications_never_generate_reply():
    assert MCPServer().handle(json.dumps({"jsonrpc": "2.0", "method": "unknown"})) is None


def test_initialized_notification_required_before_tools():
    server = MCPServer()
    server.handle(request("initialize", {"protocolVersion": "2024-11-05"}))
    reply = response(server.handle(request("tools/list")))
    assert error_code(reply) == -32002


@pytest.mark.parametrize(
    "response",
    [
        {"jsonrpc": "2.0", "id": 999, "result": {}},
        {"jsonrpc": "2.0", "id": 1, "result": {}, "error": {"code": -1, "message": "bad"}},
        {"jsonrpc": "2.0", "id": 1},
    ],
)
def test_client_rejects_malformed_or_mismatched_fake_transport(response: JSONObject) -> None:
    class FakeTransport:
        def exchange(self, wire: str) -> str:
            return json.dumps(response)

    with pytest.raises(ProtocolError):
        MCPClient(FakeTransport()).request("tools/list")


def test_client_checks_tool_result_contract():
    class FakeTransport:
        def exchange(self, wire: str) -> str:
            frame = json.loads(wire)
            return json.dumps({"jsonrpc": "2.0", "id": frame["id"], "result": {"content": 7}})

    with pytest.raises(ProtocolError):
        MCPClient(FakeTransport()).call_tool("add", {"a": 1, "b": 2})


def test_a2a_submit_get_work_complete_and_terminal_immutability():
    server = A2AServer()
    client = A2AClient(LocalTransport(server))
    task = client.submit("ticket-1", "Summarize public guide")
    assert task["status"]["state"] == "submitted"
    assert client.advance("ticket-1")["status"]["state"] == "working"
    finished = client.advance("ticket-1")
    assert finished["status"]["state"] == "completed"
    assert finished["artifacts"][0]["text"] == "Use public fixtures only."
    assert client.get("ticket-1") == finished
    with pytest.raises(ProtocolError):
        client.cancel("ticket-1")


def test_a2a_replay_is_idempotent_and_conflicting_identity_is_rejected():
    client = A2AClient(LocalTransport(A2AServer()))
    assert client.submit("same", "Read guide") == client.submit("same", "Read guide")
    with pytest.raises(ProtocolError):
        client.submit("same", "Different task")
    assert client.cancel("same")["status"]["state"] == "canceled"
    with pytest.raises(ProtocolError):
        client.advance("same")


def test_a2a_unknown_and_capacity_limits():
    client = A2AClient(LocalTransport(A2AServer(max_tasks=1)))
    with pytest.raises(ProtocolError):
        client.get("missing")
    client.submit("one", "Read guide")
    with pytest.raises(ProtocolError):
        client.submit("two", "Read guide")


def test_mcp_budget_charges_invalid_calls_and_prevents_execution():
    server = MCPServer(max_calls=1)
    client = MCPClient(LocalTransport(server))
    client.initialize()
    with pytest.raises(ProtocolError) as invalid:
        client.call_tool("shell", {})
    assert invalid.value.code == -32602
    assert server.calls == 1
    with pytest.raises(ProtocolError) as exhausted:
        client.call_tool("add", {"a": 1, "b": 2})
    assert exhausted.value.code == -32000
    assert server.calls == 1


def test_stdio_oversize_frame_is_drained_before_next_request():
    output = io.StringIO()
    incoming = io.StringIO(
        "x" * 70_000 + "\n" + request("initialize", {"protocolVersion": "2024-11-05"}) + "\n"
    )
    serve_lines(MCPServer(), incoming, output)
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert replies[0]["error"]["code"] == -32700
    assert replies[1]["result"]["protocolVersion"] == "2024-11-05"


def test_mcp_repeated_initialization_and_positional_params_rejected():
    client, _ = mcp_pair()
    with pytest.raises(ProtocolError):
        client.initialize()
    server = MCPServer()
    reply = response(
        server.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": []}))
    )
    assert error_code(reply) == -32602


def test_a2a_failed_tasks_are_terminal_and_results_are_copied():
    client = A2AClient(LocalTransport(A2AServer()))
    task = client.submit("no-guide", "unsupported task")
    task["status"]["state"] = "completed"
    assert client.get("no-guide")["status"]["state"] == "submitted"
    assert client.advance("no-guide")["status"]["state"] == "working"
    assert client.advance("no-guide")["status"]["state"] == "failed"
    with pytest.raises(ProtocolError):
        client.advance("no-guide")


@pytest.mark.parametrize("state", [{}, [], "invented"])
def test_a2a_client_rejects_malformed_status_from_fake_transport(state: object) -> None:
    class FakeTransport:
        def exchange(self, wire: str) -> str:
            frame = json.loads(wire)
            return json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": frame["id"],
                    "result": {"id": "one", "status": {"state": state}, "artifacts": []},
                }
            )

    with pytest.raises(ProtocolError):
        A2AClient(FakeTransport()).get("one")


def test_adk_import_is_lazy_and_builds_real_api_shape_with_fake_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No google-adk package is needed for the offline contract test.
    class Captured:
        model = ""

        def __init__(self) -> None:
            self.model = ""
            self.tools: list[Callable[[str], JSONObject]] = []

    captured = Captured()

    class Agent:
        def __init__(self, **kwargs: object) -> None:
            model = kwargs.get("model")
            tools = kwargs.get("tools")
            if not isinstance(model, str) or not isinstance(tools, list):
                raise AssertionError("unexpected Agent arguments")
            captured.model = model
            captured.tools = cast(list[Callable[[str], JSONObject]], tools)

    monkeypatch.setitem(sys.modules, "google.adk.agents", types.SimpleNamespace(Agent=Agent))
    agent = build_agent(model="explicit-test-model")
    assert isinstance(agent, Agent)
    assert captured.model == "explicit-test-model"
    assert captured.tools[0]("guide")["status"] == "success"
    assert captured.tools[0]("private")["status"] == "error"
