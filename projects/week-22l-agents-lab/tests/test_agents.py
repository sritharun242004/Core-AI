"""Boundary tests precede the controller implementation; all tools are local fixtures."""

import threading
from collections.abc import Mapping
from typing import NoReturn, cast

import pytest
from agents_lab import (
    Action,
    BoundedMemory,
    Budget,
    Call,
    PlanAndExecute,
    ReAct,
    Reflexion,
    ScriptedModel,
    Tool,
    ToolRegistry,
    fixture_registry,
)
from agents_lab.tools import Observation, Primitive


def test_react_observes_then_finishes_and_records_audit():
    model = ScriptedModel([Action(calls=(Call("add", {"a": 2, "b": 3}),)), Action(final="5")])
    result = ReAct(fixture_registry(), model, Budget(3, 2)).run("add two and three")
    assert (result.status, result.answer, result.steps, result.calls) == ("completed", "5", 2, 1)
    assert model.contexts[1].observations[0].value == 5
    assert [event.kind for event in result.trajectory] == ["action", "observation", "action"]


@pytest.mark.parametrize(
    "name,args,code",
    [
        ("shell", {"command": "echo no"}, "unknown_tool"),
        ("lookup", {"key": "private"}, "tool_error"),
        ("add", {"a": True, "b": 2}, "invalid_arguments"),
        ("add", {"a": float("nan"), "b": 2}, "invalid_arguments"),
        ("add", {"a": 1001, "b": 2}, "invalid_arguments"),
        ("add", {"a": 1, "b": 2, "extra": 3}, "invalid_arguments"),
        ("add", {"a": 1}, "invalid_arguments"),
    ],
)
def test_schema_and_capability_denials_are_counted(
    name: str, args: Mapping[str, Primitive], code: str
):
    budget = Budget(2, 1)
    observation = fixture_registry().dispatch((Call(name, args),), budget)[0]
    assert observation.error == code
    assert budget.calls == 1


def test_batch_reservation_prevents_partial_execution():
    budget = Budget(3, 1)
    result = fixture_registry().dispatch((Call("add", {"a": 1, "b": 1}),) * 2, budget)
    assert all(item.error == "call_budget" for item in result)
    assert budget.calls == 0


def test_parallel_independent_reads_join_in_input_order():
    barrier = threading.Barrier(2)

    def read(args: Mapping[str, Primitive]) -> object:
        barrier.wait(timeout=3)
        return args["key"]

    registry = ToolRegistry()
    registry.register(Tool("read", {"key": str}, str, read))
    budget = Budget(2, 2)
    result = registry.dispatch(
        (Call("read", {"key": "first"}), Call("read", {"key": "second"})), budget, parallel=True
    )
    assert [item.value for item in result] == ["first", "second"]


def test_parallel_dependencies_and_writes_rejected_before_execution():
    seen: list[int] = []
    registry = ToolRegistry()
    registry.register(Tool("write", {}, str, lambda args: seen.append(1) or "ok", read_only=False))
    assert (
        registry.dispatch((Call("write", {}),), Budget(2, 2), parallel=True)[0].error
        == "not_read_only"
    )
    result = fixture_registry().dispatch(
        (Call("lookup", {"key": "guide"}, depends_on=("prior",)),), Budget(2, 2), parallel=True
    )
    assert result[0].error == "dependent_batch"
    assert not seen


def test_registry_rejects_duplicates_and_validates_tool_output():
    registry = ToolRegistry()
    tool = Tool("broken", {}, str, lambda args: 9)
    registry.register(tool)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(tool)
    assert registry.dispatch((Call("broken", {}),), Budget(1, 1))[0].error == "invalid_output"


def test_tool_exceptions_do_not_leak_exception_text():
    def broken(args: Mapping[str, Primitive]) -> NoReturn:
        raise RuntimeError("private fake token")

    registry = ToolRegistry()
    registry.register(Tool("broken", {}, str, broken))
    result = registry.dispatch((Call("broken", {}),), Budget(1, 1))[0]
    assert result.error == "tool_error"
    assert result.value is None


@pytest.mark.parametrize(
    "action",
    [Action(), Action(final="x", calls=(Call("add", {"a": 1, "b": 2}),)), Action(final="x" * 4097)],
)
def test_structured_action_validation(action: Action):
    result = ReAct(fixture_registry(), ScriptedModel([action]), Budget(2, 2)).run("test")
    assert result.status == "invalid_action"
    assert result.steps == 1
    assert result.calls == 0


def test_step_and_call_limits_are_global_and_stop_before_extra_model_step():
    action = Action(calls=(Call("add", {"a": 1, "b": 2}),))
    model = ScriptedModel([action] * 5)
    result = ReAct(fixture_registry(), model, Budget(2, 5)).run("repeat")
    assert (result.status, result.steps, result.calls) == ("step_budget", 2, 2)
    model = ScriptedModel([action] * 5)
    result = ReAct(fixture_registry(), model, Budget(5, 1)).run("repeat")
    assert (result.status, result.steps, result.calls) == ("call_budget", 2, 1)


def test_model_exhaustion_is_explicit():
    result = ReAct(fixture_registry(), ScriptedModel([]), Budget(2, 2)).run("test")
    assert result.status == "model_exhausted"
    assert result.steps == 1


def test_memory_is_bounded_and_passed_as_data_not_authority():
    memory = BoundedMemory(capacity=2, max_chars=10)
    for note in ["first", "second", "third"]:
        memory.add(note)
    assert memory.snapshot() == ("second", "third")
    with pytest.raises(ValueError):
        memory.add("x" * 11)
    model = ScriptedModel([Action(final="ok")])
    ReAct(fixture_registry(), model, Budget(1, 1), memory).run("test")
    assert model.contexts[0].memory == ("second", "third")


def test_reflexion_retries_failed_answer_with_feedback_without_resetting_budget():
    model = ScriptedModel([Action(final="4"), Action(final="5")])
    result = Reflexion(fixture_registry(), model, Budget(3, 2)).run(
        "2+3", verifier=lambda answer: answer == "5", feedback="Recheck arithmetic", attempts=2
    )
    assert result.answer == "5"
    assert result.steps == 2
    assert model.contexts[1].memory == ("Recheck arithmetic",)
    assert "reflection" in [event.kind for event in result.trajectory]


def test_reflexion_cannot_turn_failed_verification_into_success():
    result = Reflexion(fixture_registry(), ScriptedModel([Action(final="bad")]), Budget(1, 1)).run(
        "test", verifier=lambda answer: False, feedback="try again", attempts=3
    )
    assert result.status == "step_budget"
    assert result.steps == 1


def test_plan_execute_uses_scripted_plan_and_shares_budget():
    model = ScriptedModel(
        [Action(final="read done"), Action(final="sum done")],
        plan=("Read public guide", "Compute total"),
    )
    result = PlanAndExecute(fixture_registry(), model, Budget(3, 1)).run("two subtasks")
    assert result.status == "completed"
    assert result.answer == "read done\nsum done"
    assert result.steps == 3  # one planning step plus two execution steps
    assert model.contexts[1].memory == ("Subtask 1: read done",)


def test_invalid_and_oversized_plans_are_rejected():
    model = ScriptedModel([], plan=("x",) * 5)
    result = PlanAndExecute(fixture_registry(), model, Budget(8, 1)).run("test", max_subtasks=4)
    assert result.status == "invalid_plan"
    assert result.steps == 1


def test_zero_budget_never_calls_model():
    model = ScriptedModel([Action(final="ok")])
    result = ReAct(fixture_registry(), model, Budget(0, 0)).run("test")
    assert result.status == "step_budget"
    assert not model.contexts


@pytest.mark.parametrize("steps,calls", [(-1, 1), (1, -1), (True, 1)])
def test_invalid_budgets(steps: int | bool, calls: int):
    with pytest.raises(ValueError):
        Budget(cast(int, steps), calls)


@pytest.mark.parametrize(
    "call",
    [
        Call(cast(str, []), {}),
        Call("x" * 65, {}),
        Call("lookup", {}, depends_on=cast(tuple[str, ...], [])),
    ]
)
def test_malformed_runtime_call_fields_are_rejected(call: Call):
    result = ReAct(fixture_registry(), ScriptedModel([Action(calls=(call,))]), Budget(1, 1)).run(
        "test"
    )
    assert result.status == "invalid_action"
    assert result.calls == 0


def test_nonboolean_parallel_flag_is_rejected():
    action = Action(calls=(Call("lookup", {"key": "guide"}),), parallel=cast(bool, "yes"))
    result = ReAct(fixture_registry(), ScriptedModel([action]), Budget(1, 1)).run("test")
    assert result.status == "invalid_action"


def test_mutating_adapter_context_cannot_corrupt_tool_history():
    args = {"a": 2, "b": 3}
    model = ScriptedModel([Action(calls=(Call("add", args),)), Action(final="5")])
    args["a"] = 999
    result = ReAct(fixture_registry(), model, Budget(2, 1)).run("test")
    observation = cast(Observation, result.trajectory[1].data)
    assert observation.value == 5


def test_registry_capacity_and_batch_size_fail_before_execution():
    registry = ToolRegistry(max_tools=1, max_batch=1)
    registry.register(Tool("one", {}, str, lambda args: "one"))
    with pytest.raises(ValueError, match="full"):
        registry.register(Tool("two", {}, str, lambda args: "two"))
    budget = Budget(1, 2)
    with pytest.raises(ValueError):
        registry.dispatch((Call("one", {}), Call("one", {})), budget)
    assert budget.calls == 0
