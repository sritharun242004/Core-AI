"""Deterministic, budgeted teaching controllers; no chain-of-thought is collected."""

from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

from .tools import Budget, Call, Observation, ToolRegistry


@dataclass(frozen=True)
class Action:
    calls: tuple[Call, ...] = ()
    final: str | None = None
    parallel: bool = False


@dataclass(frozen=True)
class Context:
    task: str
    observations: tuple[Observation, ...]
    memory: tuple[str, ...]


@dataclass(frozen=True)
class Event:
    kind: str
    data: object


@dataclass(frozen=True)
class RunResult:
    status: str
    answer: str | None
    steps: int
    calls: int
    trajectory: tuple[Event, ...]


class Model(Protocol):
    def act(self, context: Context) -> Action: ...
    def plan(self, task: str) -> tuple[str, ...]: ...


class ScriptedModel:
    """A test adapter, not an LLM or a natural-language planning implementation."""

    def __init__(self, actions: list[Action], *, plan: tuple[str, ...] = ()):
        self._actions = iter(deepcopy(actions))
        self._plan = plan
        self.contexts: list[Context] = []

    def act(self, context: Context) -> Action:
        self.contexts.append(deepcopy(context))
        return next(self._actions)

    def plan(self, task: str) -> tuple[str, ...]:
        return self._plan


class BoundedMemory:
    """FIFO feedback/subtask summaries, never authoritative instructions."""

    def __init__(self, capacity: int = 8, max_chars: int = 4096):
        if type(capacity) is not int or type(max_chars) is not int:
            raise ValueError("memory limits must be integers")
        if not 1 <= capacity <= 128 or not 1 <= max_chars <= 8192:
            raise ValueError("invalid memory limits")
        self._notes: deque[str] = deque(maxlen=capacity)
        self.max_chars = max_chars

    def add(self, note: str) -> None:
        if not isinstance(note, str) or len(note) > self.max_chars:
            raise ValueError("invalid memory note")
        self._notes.append(note)

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self._notes)


class ReAct:
    """Observe → choose typed action → execute → observe, bounded by a host budget.

    Execution traces contain decisions/results, not claims about internal reasoning.
    Controller instances are one task/session; retry controllers share their budget.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        model: Model,
        budget: Budget,
        memory: BoundedMemory | None = None,
    ):
        self.registry = registry
        self.model = model
        self.budget = budget
        self.memory = memory if memory is not None else BoundedMemory()
        self.events: list[Event] = []

    def _result(self, status: str, answer: str | None = None) -> RunResult:
        return RunResult(
            status, answer, self.budget.steps, self.budget.calls, tuple(deepcopy(self.events))
        )

    def _execute(self, task: str) -> RunResult:
        observations: tuple[Observation, ...] = ()
        while self.budget.step():
            try:
                action = self.model.act(Context(task, observations, self.memory.snapshot()))
            except StopIteration:
                return self._result("model_exhausted")
            except Exception:
                return self._result("model_error")
            if not isinstance(action, Action):
                return self._result("invalid_action")
            valid_final = isinstance(action.final, str) and len(action.final) <= 4096
            if (action.final is not None and (not valid_final or action.calls)) or (
                action.final is None and not action.calls
            ):
                return self._result("invalid_action")
            self.events.append(Event("action", deepcopy(action)))
            if action.final is not None:
                return self._result("completed", action.final)
            try:
                observations = self.registry.dispatch(
                    action.calls, self.budget, parallel=action.parallel
                )
            except (ValueError, TypeError):
                return self._result("invalid_action")
            self.events.extend(Event("observation", item) for item in observations)
            if any(item.error == "call_budget" for item in observations):
                return self._result("call_budget")
        return self._result("step_budget")

    def run(self, task: str) -> RunResult:
        if not isinstance(task, str) or not 1 <= len(task) <= 8192:
            raise ValueError("task must contain 1..8192 characters")
        return self._execute(task)


class Reflexion(ReAct):
    """Verifier-guided retries with bounded feedback memory.

    Feedback is supplied by a trusted external evaluator in this lab, not generated
    by an LLM. The control pattern illustrates Reflexion; it is not a paper reproduction.
    """

    def run(
        self, task: str, *, verifier: Callable[[str], bool], feedback: str, attempts: int = 2
    ) -> RunResult:
        if type(attempts) is not int or not 1 <= attempts <= 16:
            raise ValueError("attempts must be in 1..16")
        if not isinstance(feedback, str) or len(feedback) > self.memory.max_chars:
            raise ValueError("feedback exceeds memory schema")
        for _ in range(attempts):
            result = super().run(task)
            if result.status != "completed":
                return result
            try:
                accepted = verifier(result.answer)
            except Exception:
                return self._result("verifier_error")
            if accepted is True:
                return result
            self.memory.add(feedback)
            self.events.append(Event("reflection", feedback))
        return self._result("verification_failed")


class PlanAndExecute(ReAct):
    """Charge one planning step, validate the plan, execute subtasks sequentially."""

    def run(self, task: str, *, max_subtasks: int = 4) -> RunResult:
        if not isinstance(task, str) or not 1 <= len(task) <= 8192:
            raise ValueError("task must contain 1..8192 characters")
        if type(max_subtasks) is not int or not 1 <= max_subtasks <= 16:
            raise ValueError("max_subtasks must be in 1..16")
        if not self.budget.step():
            return self._result("step_budget")
        try:
            plan = self.model.plan(task)
        except Exception:
            return self._result("model_error")
        if (
            not isinstance(plan, tuple)
            or not 1 <= len(plan) <= max_subtasks
            or any(not isinstance(item, str) or not 1 <= len(item) <= 1024 for item in plan)
        ):
            return self._result("invalid_plan")
        self.events.append(Event("plan", plan))
        answers = []
        for index, subtask in enumerate(plan):
            result = self._execute(f"{task}\nSubtask {index + 1}: {subtask}")
            if result.status != "completed":
                return result
            answers.append(result.answer)
            note = f"Subtask {index + 1}: {result.answer}"
            self.memory.add(note[: self.memory.max_chars])
        return self._result("completed", "\n".join(answers))
