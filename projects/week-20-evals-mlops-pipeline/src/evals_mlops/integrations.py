"""Explicit optional exports. No integration package is imported automatically."""

from pathlib import Path

from .evaluation import Example, RunManifest


def export_mlflow(manifest: RunManifest, metrics: dict[str, float], directory: str | Path) -> None:
    """Write to a caller-selected local file store; never a remote tracking URI."""
    import mlflow

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


def _local_run(mlflow, target):
    mlflow.set_tracking_uri(target.as_uri())
    mlflow.set_experiment("core-ai-offline")
    return mlflow.start_run(run_name="evaluation")


def inspect_task(examples: list[Example]):
    """Build a genuine Inspect task; constructing it does not call a model.

    Execution requires a separately configured local/provider model. Inspect AI
    is UK AI Security Institute tooling. This is not MT-Bench or SWE-Bench.
    """
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.scorer import exact
    from inspect_ai.solver import generate

    return Task(
        dataset=MemoryDataset(
            [Sample(input=e.prompt, target=e.expected, id=e.id) for e in examples]
        ),
        solver=generate(),
        scorer=exact(),
    )
