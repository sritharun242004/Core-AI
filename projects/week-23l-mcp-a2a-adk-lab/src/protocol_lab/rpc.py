"""Bounded newline JSON-RPC transport primitives, not a general RPC implementation."""

import json
from typing import Any, Protocol, TextIO

MAX_FRAME = 65_536


class ProtocolError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def encode(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def decode(wire: str) -> Any:
    def reject_constant(value):
        raise ValueError("nonfinite JSON number")

    if not isinstance(wire, str) or len(wire) > MAX_FRAME:
        raise ValueError("frame exceeds character limit")
    return json.loads(wire, parse_constant=reject_constant)


def error_frame(ident: Any, code: int, message: str) -> str:
    return encode({"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}})


class JSONRPCServer:
    """Requests only, plus no-response notifications; batch frames are not supported."""

    def dispatch(self, method: str, params: dict[str, Any], *, notification: bool) -> Any:
        raise ProtocolError(-32601, "method not found")

    def handle(self, wire: str) -> str | None:
        try:
            frame = decode(wire)
        except (ValueError, TypeError, RecursionError):
            return error_frame(None, -32700, "parse error or frame limit")
        if not isinstance(frame, dict) or frame.get("jsonrpc") != "2.0":
            return error_frame(None, -32600, "invalid request")
        ident = frame.get("id")
        if not isinstance(frame.get("method"), str) or (
            "id" in frame and type(ident) not in (int, str)
        ):
            return error_frame(None, -32600, "invalid request")
        notification = "id" not in frame
        params = frame.get("params", {})
        try:
            if not isinstance(params, dict):
                raise ProtocolError(-32602, "params must be an object")
            result = self.dispatch(frame["method"], params, notification=notification)
            return (
                None if notification else encode({"jsonrpc": "2.0", "id": ident, "result": result})
            )
        except ProtocolError as error:
            return None if notification else error_frame(ident, error.code, str(error))
        except Exception:
            return None if notification else error_frame(ident, -32603, "internal error")


class Transport(Protocol):
    def exchange(self, wire: str) -> str | None: ...


class LocalTransport:
    """Serialized in-process loopback; no direct sharing of request/result objects."""

    def __init__(self, server: JSONRPCServer):
        self.server = server
        self.requests: list[str] = []

    def exchange(self, wire: str) -> str | None:
        # A small bounded recorder, not a production audit-log persistence service.
        self.requests.append(wire)
        self.requests = self.requests[-128:]
        return self.server.handle(wire)


class RPCClient:
    def __init__(self, transport: Transport):
        self.transport = transport
        self._next_id = 0

    def notify(self, method: str, params: dict | None = None) -> None:
        response = self.transport.exchange(
            encode(
                {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params or {},
                }
            )
        )
        if response is not None:
            raise ProtocolError(-32600, "notification unexpectedly received a reply")

    def request(self, method: str, params: dict | None = None) -> Any:
        self._next_id += 1
        wire = encode(
            {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
        )
        response = self.transport.exchange(wire)
        try:
            frame = decode(response)
        except (ValueError, TypeError, RecursionError) as error:
            raise ProtocolError(-32600, "invalid response") from error
        if (
            not isinstance(frame, dict)
            or frame.get("jsonrpc") != "2.0"
            or (type(frame.get("id")) is not int or frame["id"] != self._next_id)
            or (("result" in frame) == ("error" in frame))
        ):
            raise ProtocolError(-32600, "invalid response envelope or id")
        if "error" in frame:
            error = frame["error"]
            if (
                not isinstance(error, dict)
                or type(error.get("code")) is not int
                or (not isinstance(error.get("message"), str))
            ):
                raise ProtocolError(-32600, "invalid error envelope")
            raise ProtocolError(error["code"], error["message"])
        return frame["result"]


def serve_lines(server: JSONRPCServer, source: TextIO, sink: TextIO) -> None:
    """One JSON object per line; diagnostics must never be printed to this stream."""
    while True:
        line = source.readline(MAX_FRAME + 2)
        if not line:
            break
        if len(line.rstrip("\n")) > MAX_FRAME:
            while line and not line.endswith("\n"):
                line = source.readline(MAX_FRAME + 2)
            reply = error_frame(None, -32700, "frame limit")
        else:
            reply = server.handle(line)
        if reply is not None:
            sink.write(reply + "\n")
            sink.flush()
