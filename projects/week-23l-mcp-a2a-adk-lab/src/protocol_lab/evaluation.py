"""Repository-shaped in-memory tasks; no arbitrary code or test commands execute.

The fixtures and deterministic validators are classroom tasks, not SWE-Bench data.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath


@dataclass(frozen=True)
class RepositoryTask:
    task_id: str
    instruction: str
    files: dict[str, str]
    editable: tuple[str, ...]
    verify: Callable[[dict[str, str]], bool]
    max_steps: int = 8
    max_calls: int = 8


@dataclass(frozen=True)
class EvaluationAttempt:
    task_id: str
    replacements: dict[str, str]
    steps: int = 0
    calls: int = 0
    denied_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self):
        counts = (self.steps, self.calls, self.denied_calls, self.input_tokens, self.output_tokens)
        if any(type(n) is not int or n < 0 for n in counts) or self.denied_calls > self.calls:
            raise ValueError("invalid trajectory counters")


@dataclass(frozen=True)
class CaseResult:
    task_id: str
    passed: bool
    safe: bool
    reason: str


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[CaseResult, ...]
    task_success: float
    safe_success: float
    mean_steps: float
    total_calls: int
    denied_attempt_rate: float
    input_tokens: int
    output_tokens: int
    suite: str = field(default="local-repository-fixtures-v1", init=False)
    scope: str = field(
        default="Synthetic offline repository tasks; not SWE-Bench or a model score", init=False
    )


def apply_replacements(task: RepositoryTask, replacements: dict[str, str]) -> dict[str, str]:
    """A bounded allowlisted whole-file patch, never an executable shell/diff command."""
    if not isinstance(replacements, dict) or len(replacements) > 8:
        raise ValueError("invalid patch")
    files = dict(task.files)
    for path, content in replacements.items():
        if not isinstance(path, str) or path not in task.editable or path not in files:
            raise ValueError("path is not editable")
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in path:
            raise ValueError("unsafe path")
        if not isinstance(content, str) or len(content) > 8192:
            raise ValueError("invalid content")
        files[path] = content
    return files


def repository_fixtures() -> list[RepositoryTask]:
    def timeout_check(files):
        data = json.loads(files["config.json"])
        return (
            data.keys() == {"timeout_seconds"}
            and type(data["timeout_seconds"]) is int
            and (data["timeout_seconds"] == 30)
            and files["src/client.py"] == 'TIMEOUT_KEY = "timeout_seconds"\n'
        )

    def readme_check(files):
        version = json.loads(files["package.json"])["version"]
        return files["README.md"] == f"# Widget\nVersion: {version}\n"

    return [
        RepositoryTask(
            "config-timeout",
            "Set the documented client timeout to 30 seconds.",
            {
                "config.json": '{"timeout_seconds": 3}',
                "src/client.py": 'TIMEOUT_KEY = "timeout_seconds"\n',
                "README.md": "The timeout is configured in config.json.\n",
            },
            ("config.json",),
            timeout_check,
        ),
        RepositoryTask(
            "readme-version",
            "Synchronize README version with package.json.",
            {"README.md": "# Widget\nVersion: 1\n", "package.json": '{"version": 2}'},
            ("README.md",),
            readme_check,
        ),
    ]


def evaluate(tasks: list[RepositoryTask], attempts: list[EvaluationAttempt]) -> EvaluationReport:
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("suite needs unique nonempty tasks")
    by_id = {attempt.task_id: attempt for attempt in attempts}
    task_ids = {task.task_id for task in tasks}
    if len(by_id) != len(attempts) or not by_id.keys() <= task_ids:
        raise ValueError("duplicate or unknown attempt")
    results = []
    for task in tasks:
        attempt = by_id.get(task.task_id)
        if attempt is None:
            results.append(CaseResult(task.task_id, False, False, "missing_attempt"))
            continue
        if attempt.steps > task.max_steps or attempt.calls > task.max_calls:
            results.append(CaseResult(task.task_id, False, False, "budget_exceeded"))
            continue
        try:
            files = apply_replacements(task, attempt.replacements)
        except ValueError:
            results.append(CaseResult(task.task_id, False, False, "invalid_patch"))
            continue
        try:
            passed = task.verify(files) is True
            reason = "passed" if passed else "verification_failed"
        except Exception:
            passed, reason = False, "verification_failed"
        results.append(
            CaseResult(task.task_id, passed, passed and not attempt.denied_calls, reason)
        )
    count = len(tasks)
    return EvaluationReport(
        tuple(results),
        sum(r.passed for r in results) / count,
        sum(r.safe for r in results) / count,
        sum(a.steps for a in attempts) / count,
        sum(a.calls for a in attempts),
        sum(a.denied_calls > 0 for a in attempts) / count,
        sum(a.input_tokens for a in attempts),
        sum(a.output_tokens for a in attempts),
    )
