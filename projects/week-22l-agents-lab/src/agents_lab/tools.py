"""Trusted local capabilities, strict small schemas, and ordered read-only fan-out.

This is not a Python sandbox. Register trusted callbacks only. No tool in the
fixture can access the filesystem, network, subprocesses, or credentials.
"""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol, cast

Primitive = int | float | str | bool


class ToolHandler(Protocol):
    def __call__(self, arguments: Mapping[str, Primitive], /) -> object: ...


@dataclass
class Budget:
    max_steps: int
    max_calls: int
    steps: int = field(default=0, init=False)
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
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
    arguments: Mapping[str, Primitive]
    depends_on: tuple[str, ...] = ()


def _runtime_value(value: object) -> object:
    return value


def _valid_call_envelope(value: object) -> bool:
    if not isinstance(value, Call):
        return False
    name = _runtime_value(value.name)
    depends_on = _runtime_value(value.depends_on)
    if not isinstance(name, str) or not 1 <= len(name) <= 64:
        return False
    if not isinstance(depends_on, tuple):
        return False
    candidate_dependencies = cast(tuple[object, ...], depends_on)
    return all(isinstance(dependency, str) for dependency in candidate_dependencies)


@dataclass(frozen=True)
class Observation:
    name: str
    value: object = None
    error: str | None = None


def matches(value: object, kind: type[Primitive]) -> bool:
    """An intentionally small schema: exact primitives, no bool-as-int, finite numbers."""
    if kind is int:
        number = cast(int, value)
        return type(value) is int and abs(number) <= 1000 and isfinite(number)
    if kind is float:
        if type(value) not in (int, float):
            return False
        number = cast(int | float, value)
        return abs(number) <= 1000 and isfinite(number)
    if kind is str:
        if type(value) is not str:
            return False
        return len(value) <= 4096
    if kind is bool:
        return type(value) is bool
    return False


@dataclass(frozen=True)
class Tool:
    name: str
    parameters: Mapping[str, type[Primitive]]
    output: type[Primitive]
    handler: ToolHandler
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
        supported: tuple[type[Primitive], ...] = (int, float, str, bool)
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
        self, calls: object, budget: Budget, *, parallel: bool = False
    ) -> tuple[Observation, ...]:
        if (
            type(parallel) is not bool
            or not isinstance(calls, tuple)
            or (not calls or len(cast(tuple[object, ...], calls)) > self.max_batch)
        ):
            raise ValueError(
                "batch must contain 1..max_batch typed calls and a boolean parallel flag"
            )
        candidate_calls = cast(tuple[object, ...], calls)
        if any(not _valid_call_envelope(call) for call in candidate_calls):
            raise ValueError("invalid call envelope")
        typed_calls = cast(tuple[Call, ...], deepcopy(candidate_calls))
        if not budget.reserve_calls(len(typed_calls)):
            return tuple(Observation(c.name, error="call_budget") for c in typed_calls)
        if any(c.depends_on for c in typed_calls):
            # Dependencies must become separate controller steps, never string substitution.
            return tuple(Observation(c.name, error="dependent_batch") for c in typed_calls)
        if parallel and any(
            c.name in self._tools and not self._tools[c.name].read_only for c in typed_calls
        ):
            return tuple(Observation(c.name, error="not_read_only") for c in typed_calls)
        if parallel:
            with ThreadPoolExecutor(max_workers=min(4, len(typed_calls))) as pool:
                return tuple(pool.map(self._invoke, typed_calls))
        return tuple(self._invoke(c) for c in typed_calls)


def fixture_registry() -> ToolRegistry:
    """Only arithmetic and an immutable public in-memory catalogue are allowlisted."""
    records = {"guide": "Read only public fixtures.", "hours": "Study for 20 hours."}

    def add(arguments: Mapping[str, Primitive]) -> object:
        left, right = arguments["a"], arguments["b"]
        if type(left) not in (int, float) or type(right) not in (int, float):
            raise TypeError("arithmetic arguments must be numeric")
        return cast(int | float, left) + cast(int | float, right)

    def lookup(arguments: Mapping[str, Primitive]) -> object:
        key = arguments["key"]
        if type(key) is not str:
            raise TypeError("lookup key must be a string")
        return records[key]

    registry = ToolRegistry()
    registry.register(Tool("add", {"a": float, "b": float}, float, add))
    registry.register(Tool("lookup", {"key": str}, str, lookup))
    return registry
