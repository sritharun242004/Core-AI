"""Trusted local capabilities, strict small schemas, and ordered read-only fan-out.

This is not a Python sandbox. Register trusted callbacks only. No tool in the
fixture can access the filesystem, network, subprocesses, or credentials.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from typing import Any


@dataclass
class Budget:
    max_steps: int
    max_calls: int
    steps: int = field(default=0, init=False)
    calls: int = field(default=0, init=False)

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.max_steps, self.max_calls)):
            raise ValueError("budgets must be nonnegative integers")

    def step(self) -> bool:
        if self.steps >= self.max_steps:
            return False
        self.steps += 1
        return True

    def reserve_calls(self, count: int) -> bool:
        if type(count) is not int or count < 0:
            raise ValueError("count must be nonnegative")
        if self.calls + count > self.max_calls:
            return False
        self.calls += count
        return True


@dataclass(frozen=True)
class Call:
    name: str
    arguments: dict[str, Any]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Observation:
    name: str
    value: Any = None
    error: str | None = None


def matches(value: Any, kind: type) -> bool:
    """An intentionally small schema: exact primitives, no bool-as-int, finite numbers."""
    if kind in (int, float):
        valid_type = type(value) is int if kind is int else type(value) in (int, float)
        return valid_type and abs(value) <= 1000 and isfinite(value)
    if kind is str:
        return type(value) is str and len(value) <= 4096
    if kind is bool:
        return type(value) is bool
    return False


@dataclass(frozen=True)
class Tool:
    name: str
    parameters: dict[str, type]
    output: type
    handler: Callable[[dict[str, Any]], Any]
    read_only: bool = True


class ToolRegistry:
    """Registration is host-only; the model can only request an existing name.

    Batch admission happens on the coordinator thread before worker dispatch.
    Registry and budget objects are intentionally not shared between controllers.
    """

    def __init__(self, max_tools: int = 16, max_batch: int = 8):
        if type(max_tools) is not int or type(max_batch) is not int:
            raise ValueError("limits must be integers")
        if not 1 <= max_tools <= 128 or not 1 <= max_batch <= 32:
            raise ValueError("invalid registry limits")
        self.max_tools = max_tools
        self.max_batch = max_batch
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError("duplicate tool")
        if len(self._tools) >= self.max_tools:
            raise ValueError("registry full")
        if not tool.name.isidentifier() or len(tool.name) > 64:
            raise ValueError("invalid tool name")
        supported = (int, float, str, bool)
        if tool.output not in supported or any(
            t not in supported for t in tool.parameters.values()
        ):
            raise ValueError("unsupported schema type")
        self._tools[tool.name] = deepcopy(tool)

    def _invoke(self, call: Call) -> Observation:
        tool = self._tools.get(call.name)
        if tool is None:
            return Observation(call.name, error="unknown_tool")
        args = call.arguments
        if not isinstance(args, dict) or args.keys() != tool.parameters.keys():
            return Observation(call.name, error="invalid_arguments")
        if any(not matches(args[key], kind) for key, kind in tool.parameters.items()):
            return Observation(call.name, error="invalid_arguments")
        try:
            value = tool.handler(deepcopy(args))
        except Exception:
            # Public observations contain stable error categories, not sensitive tracebacks.
            return Observation(call.name, error="tool_error")
        if not matches(value, tool.output):
            return Observation(call.name, error="invalid_output")
        return Observation(call.name, deepcopy(value))

    def dispatch(
        self, calls: tuple[Call, ...], budget: Budget, *, parallel: bool = False
    ) -> tuple[Observation, ...]:
        if (
            type(parallel) is not bool
            or not isinstance(calls, tuple)
            or (not calls or len(calls) > self.max_batch)
        ):
            raise ValueError(
                "batch must contain 1..max_batch typed calls and a boolean parallel flag"
            )
        if any(
            not isinstance(c, Call)
            or not isinstance(c.name, str)
            or not 1 <= len(c.name) <= 64
            or not isinstance(c.depends_on, tuple)
            or any(not isinstance(dep, str) for dep in c.depends_on)
            for c in calls
        ):
            raise ValueError("invalid call envelope")
        calls = deepcopy(calls)
        if not budget.reserve_calls(len(calls)):
            return tuple(Observation(c.name, error="call_budget") for c in calls)
        if any(c.depends_on for c in calls):
            # Dependencies must become separate controller steps, never string substitution.
            return tuple(Observation(c.name, error="dependent_batch") for c in calls)
        if parallel and any(
            c.name in self._tools and not self._tools[c.name].read_only for c in calls
        ):
            return tuple(Observation(c.name, error="not_read_only") for c in calls)
        if parallel:
            with ThreadPoolExecutor(max_workers=min(4, len(calls))) as pool:
                return tuple(pool.map(self._invoke, calls))
        return tuple(self._invoke(c) for c in calls)


def fixture_registry() -> ToolRegistry:
    """Only arithmetic and an immutable public in-memory catalogue are allowlisted."""
    records = {"guide": "Read only public fixtures.", "hours": "Study for 20 hours."}
    registry = ToolRegistry()
    registry.register(
        Tool("add", {"a": float, "b": float}, float, lambda args: args["a"] + args["b"])
    )
    registry.register(Tool("lookup", {"key": str}, str, lambda args: records[args["key"]]))
    return registry
