"""MCP 2024-11-05-inspired teaching subset: initialize + tools/list + tools/call.

MCP originated at Anthropic. This is not a complete conformance implementation:
no resources, prompts, sampling, pagination, cancellation, auth or HTTP transport.
"""

from copy import deepcopy
from math import isfinite
from typing import cast

from .rpc import JSONObject, JSONRPCServer, ProtocolError, RPCClient

PROTOCOL_VERSION = "2024-11-05"


def object_schema(properties: JSONObject) -> JSONObject:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


NUMBER: JSONObject = {"type": "number", "minimum": -1000, "maximum": 1000}
TOOLS: list[JSONObject] = [
    {
        "name": "add",
        "description": "Add two finite bounded public numbers.",
        "inputSchema": object_schema({"a": NUMBER, "b": NUMBER}),
    },
    {
        "name": "lookup",
        "description": "Read a public in-memory fixture record.",
        "inputSchema": object_schema({"key": {"type": "string", "maxLength": 128}}),
    },
]


def validate_arguments(schema: JSONObject, arguments: JSONObject) -> None:
    """Deliberately just this declared schema subset, not a JSON Schema validator."""
    properties = schema.get("properties")
    if not isinstance(properties, dict) or arguments.keys() != properties.keys():
        raise ProtocolError(-32602, "invalid argument keys")
    for key, spec in properties.items():
        if not isinstance(spec, dict):
            raise ProtocolError(-32602, "invalid argument schema")
        value = arguments[key]
        if spec.get("type") == "number":
            if type(value) not in (int, float):
                raise ProtocolError(-32602, "number must be finite and within bounds")
            number = cast(int | float, value)
            if not -1000 <= number <= 1000 or not isfinite(number):
                raise ProtocolError(-32602, "number must be finite and within bounds")
        else:
            max_length = spec.get("maxLength")
            if type(value) is not str or type(max_length) is not int or len(value) > max_length:
                raise ProtocolError(-32602, "invalid string argument")


class MCPServer(JSONRPCServer):
    def __init__(self, max_calls: int = 64):
        if type(max_calls) is not int or not 0 <= max_calls <= 10_000:
            raise ValueError("invalid call budget")
        self.state = "new"
        self.max_calls = max_calls
        self.calls = 0
        self._records = {"guide": "Use public fixtures only.", "version": "1"}

    def dispatch(self, method: str, params: JSONObject, *, notification: bool) -> object:
        if notification:
            if method == "notifications/initialized" and self.state == "initializing":
                self.state = "ready"
            return None
        if method == "initialize":
            if self.state != "new":
                raise ProtocolError(-32600, "session already initialized")
            if params.get("protocolVersion") != PROTOCOL_VERSION:
                raise ProtocolError(-32602, "teaching server supports only " + PROTOCOL_VERSION)
            self.state = "initializing"
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "core-ai-mcp-teaching-subset", "version": "0.1.0"},
            }
        if self.state != "ready":
            raise ProtocolError(-32002, "initialize and initialized notification required")
        if method == "tools/list":
            if params:
                raise ProtocolError(-32602, "pagination not implemented")
            return {"tools": deepcopy(TOOLS)}
        if method != "tools/call":
            raise ProtocolError(-32601, "method not found")
        if self.calls >= self.max_calls:
            raise ProtocolError(-32000, "call budget exceeded")
        self.calls += 1
        if params.keys() != {"name", "arguments"} or not isinstance(params.get("name"), str):
            raise ProtocolError(-32602, "tool call needs name and arguments")
        tool_name = params.get("name")
        arguments = params.get("arguments")
        if not isinstance(tool_name, str) or not isinstance(arguments, dict):
            raise ProtocolError(-32602, "tool call needs name and arguments")
        tool = next((item for item in TOOLS if item.get("name") == tool_name), None)
        if tool is None:
            raise ProtocolError(-32602, "unknown tool")
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict):
            raise ProtocolError(-32602, "invalid tool schema")
        validate_arguments(schema, arguments)
        is_error = False
        if tool_name == "add":
            first, second = arguments.get("a"), arguments.get("b")
            if type(first) not in (int, float) or type(second) not in (int, float):
                raise ProtocolError(-32602, "invalid number arguments")
            text = str(cast(int | float, first) + cast(int | float, second))
        else:
            key = arguments.get("key")
            if not isinstance(key, str):
                raise ProtocolError(-32602, "invalid string argument")
            text = self._records.get(key)
            if text is None:
                text, is_error = "public record not found", True
        return {"content": [{"type": "text", "text": text}], "isError": is_error}


class MCPClient(RPCClient):
    def initialize(self) -> JSONObject:
        result = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "core-ai-teaching-client", "version": "0.1.0"},
            },
        )
        if (
            not isinstance(result, dict)
            or result.get("protocolVersion") != PROTOCOL_VERSION
            or (
                not isinstance(result.get("capabilities"), dict)
                or not isinstance(result.get("serverInfo"), dict)
            )
        ):
            raise ProtocolError(-32600, "invalid initialize result")
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[JSONObject]:
        result = self.request("tools/list")
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            raise ProtocolError(-32600, "invalid tools list")
        for tool in tools:
            if not isinstance(tool, dict):
                raise ProtocolError(-32600, "invalid tool descriptor")
            schema = tool.get("inputSchema")
            if not isinstance(tool.get("name"), str) or not isinstance(schema, dict):
                raise ProtocolError(-32600, "invalid tool descriptor")
            if schema.get("type") != "object":
                raise ProtocolError(-32600, "invalid tool descriptor")
        return [cast(JSONObject, tool) for tool in tools]

    def call_tool(self, name: str, arguments: JSONObject) -> JSONObject:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content") if isinstance(result, dict) else None
        if not isinstance(result, dict) or type(result.get("isError")) is not bool:
            raise ProtocolError(-32600, "invalid tool result")
        if not isinstance(content, list):
            raise ProtocolError(-32600, "invalid tool result")
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text":
                raise ProtocolError(-32600, "invalid text content")
            text = item.get("text")
            if not isinstance(text, str) or len(text) > 4096:
                raise ProtocolError(-32600, "invalid text content")
        return result
