"""Explicit optional exports. No integration package is imported automatically."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from .evaluation import Example, RunManifest


@runtime_checkable
class MLflow(Protocol):
    """Only the local-tracking surface consumed by this adapter."""

    def set_tracking_uri(self, uri: str) -> None: ...
    def set_experiment(self, experiment_name: str) -> object: ...
    def start_run(self, *, run_name: str) -> AbstractContextManager[object]: ...
    def log_params(self, params: dict[str, str | int]) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...


@runtime_checkable
class InspectTask(Protocol):
    """Opaque SDK components remain owned by Inspect, not this offline lab."""

    @property
    def dataset(self) -> object: ...
    @property
    def solver(self) -> object: ...
    @property
    def scorer(self) -> object: ...


class SampleFactory(Protocol):
    def __call__(self, *, input: str, target: str, id: str) -> object: ...


class TaskFactory(Protocol):
    def __call__(self, *, dataset: object, solver: object, scorer: object) -> object: ...


def _factory(module: str, name: str) -> object:
    value: object = getattr(import_module(module), name)
    if not callable(value):
        raise TypeError(f"{module}.{name} must be callable")
    return value


def export_mlflow(manifest: RunManifest, metrics: dict[str, float], directory: str | Path) -> None:
    """Write to a caller-selected local file store; never a remote tracking URI."""
    mlflow = import_module("mlflow")
    if not isinstance(mlflow, MLflow):
        raise TypeError("MLflow does not expose the required local tracking interface")
    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    with _local_run(mlflow, target):
        mlflow.log_params(
            {
                "model": manifest.model,
                "dataset": manifest.dataset,
                "evaluator": manifest.evaluator,
                "seed": manifest.seed,
            }
        )
        mlflow.log_metrics(metrics)


def _local_run(mlflow: MLflow, target: Path) -> AbstractContextManager[object]:
    mlflow.set_tracking_uri(target.as_uri())
    mlflow.set_experiment("core-ai-offline")
    return mlflow.start_run(run_name="evaluation")


def inspect_task(examples: list[Example]) -> InspectTask:
    """Build a genuine Inspect task; constructing it does not call a model.

    Execution requires a separately configured local/provider model. Inspect AI
    is UK AI Security Institute tooling. This is not MT-Bench or SWE-Bench.
    """
    # These casts describe just the documented constructor signatures. Callable
    # presence and the resulting task's structure are checked at the SDK boundary.
    task = cast(TaskFactory, _factory("inspect_ai", "Task"))
    dataset = cast(
        Callable[[list[object]], object], _factory("inspect_ai.dataset", "MemoryDataset")
    )
    sample = cast(SampleFactory, _factory("inspect_ai.dataset", "Sample"))
    exact = cast(Callable[[], object], _factory("inspect_ai.scorer", "exact"))
    generate = cast(Callable[[], object], _factory("inspect_ai.solver", "generate"))
    result = task(
        dataset=dataset([sample(input=e.prompt, target=e.expected, id=e.id) for e in examples]),
        solver=generate(),
        scorer=exact(),
    )
    if not isinstance(result, InspectTask):
        raise TypeError("Inspect Task does not expose dataset, solver and scorer")
    return result
