"""MCP 2024-11-05-inspired teaching subset: initialize + tools/list + tools/call.

MCP originated at Anthropic. This is not a complete conformance implementation:
no resources, prompts, sampling, pagination, cancellation, auth or HTTP transport.
"""

from copy import deepcopy
from math import isfinite

from .rpc import JSONRPCServer, ProtocolError, RPCClient

PROTOCOL_VERSION = "2024-11-05"


def object_schema(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


NUMBER = {"type": "number", "minimum": -1000, "maximum": 1000}
TOOLS = [
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


def validate_arguments(schema: dict, arguments: dict) -> None:
    """Deliberately just this declared schema subset, not a JSON Schema validator."""
    properties = schema["properties"]
    if not isinstance(arguments, dict) or arguments.keys() != properties.keys():
        raise ProtocolError(-32602, "invalid argument keys")
    for key, spec in properties.items():
        value = arguments[key]
        if spec["type"] == "number":
            if type(value) not in (int, float) or not -1000 <= value <= 1000 or not isfinite(value):
                raise ProtocolError(-32602, "number must be finite and within bounds")
        elif type(value) is not str or len(value) > spec["maxLength"]:
            raise ProtocolError(-32602, "invalid string argument")


class MCPServer(JSONRPCServer):
    def __init__(self, max_calls: int = 64):
        if type(max_calls) is not int or not 0 <= max_calls <= 10_000:
            raise ValueError("invalid call budget")
        self.state = "new"
        self.max_calls = max_calls
        self.calls = 0
        self._records = {"guide": "Use public fixtures only.", "version": "1"}

    def dispatch(self, method: str, params: dict, *, notification: bool):
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
        tool = next((item for item in TOOLS if item["name"] == params["name"]), None)
        if tool is None:
            raise ProtocolError(-32602, "unknown tool")
        arguments = params["arguments"]
        validate_arguments(tool["inputSchema"], arguments)
        is_error = False
        if tool["name"] == "add":
            text = str(arguments["a"] + arguments["b"])
        else:
            text = self._records.get(arguments["key"])
            if text is None:
                text, is_error = "public record not found", True
        return {"content": [{"type": "text", "text": text}], "isError": is_error}


class MCPClient(RPCClient):
    def initialize(self) -> dict:
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

    def list_tools(self) -> list[dict]:
        result = self.request("tools/list")
        if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
            raise ProtocolError(-32600, "invalid tools list")
        for tool in result["tools"]:
            if (
                not isinstance(tool, dict)
                or not isinstance(tool.get("name"), str)
                or (
                    not isinstance(tool.get("inputSchema"), dict)
                    or tool["inputSchema"].get("type") != "object"
                )
            ):
                raise ProtocolError(-32600, "invalid tool descriptor")
        return result["tools"]

    def call_tool(self, name: str, arguments: dict) -> dict:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        if (
            not isinstance(result, dict)
            or type(result.get("isError")) is not bool
            or (not isinstance(result.get("content"), list))
        ):
            raise ProtocolError(-32600, "invalid tool result")
        for item in result["content"]:
            if (
                not isinstance(item, dict)
                or item.get("type") != "text"
                or (not isinstance(item.get("text"), str) or len(item["text"]) > 4096)
            ):
                raise ProtocolError(-32600, "invalid text content")
        return result
