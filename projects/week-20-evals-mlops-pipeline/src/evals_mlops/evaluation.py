"""Small reproducible evaluation building blocks, not a benchmark leaderboard."""

import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

FloatValues = Sequence[float] | NDArray[np.float64]


class JudgeResult(TypedDict):
    a_score: float
    consistent: bool
    calls: int


class RegressionResult(TypedDict):
    passed: bool
    mean_delta: float
    delta_interval: tuple[float, float]


class Trajectory(TypedDict):
    success: bool
    tool_calls: int
    unsafe_attempts: int


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    expected: str


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def dataset_digest(examples: Sequence[Example]) -> str:
    """Content hash includes IDs, prompts and labels, but not input row order."""
    if len({e.id for e in examples}) != len(examples):
        raise ValueError("example IDs must be unique")
    return sha256(
        _canonical([asdict(e) for e in sorted(examples, key=lambda e: e.id)]).encode()
    ).hexdigest()


def split_dataset(
    examples: Sequence[Example],
    *,
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> tuple[tuple[Example, ...], ...]:
    """Seeded random split; use chronological/group splits when domain requires it."""
    dataset_digest(examples)
    if (
        len(fractions) != 3
        or any(not math.isfinite(x) or x <= 0 for x in fractions)
        or not math.isclose(sum(fractions), 1.0)
    ):
        raise ValueError("three positive fractions must sum to one")
    ordered = sorted(examples, key=lambda e: e.id)
    indices = np.random.default_rng(seed).permutation(len(ordered))
    n_train, n_val = int(len(ordered) * fractions[0]), int(len(ordered) * fractions[1])
    if min(n_train, n_val, len(ordered) - n_train - n_val) < 1:
        raise ValueError("each split needs at least one example")
    return tuple(
        tuple(ordered[int(i)] for i in chunk)
        for chunk in np.split(indices, [n_train, n_train + n_val])
    )


def _values(values: FloatValues) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("values must be a nonempty finite vector")
    return array


def exact_match(predictions: Sequence[str], expected: Sequence[str]) -> float:
    if not predictions or len(predictions) != len(expected):
        raise ValueError("prediction and target lengths must match and be nonempty")

    def normalize(s: str) -> str:
        return " ".join(s.casefold().split())

    return sum(
        normalize(a) == normalize(b) for a, b in zip(predictions, expected, strict=True)
    ) / len(expected)


def judge_pair(
    prompt: str, a: str, b: str, judge: Callable[[str, str, str], object]
) -> JudgeResult:
    """Two rubric-judge calls; swap positions, map scores back to original A.

    A contradictory pair averages to .5 but is NOT marked as an agreed tie.
    Adapters must treat candidate text as untrusted data, not instructions.
    """
    scores: list[float] = []
    for left, right, swapped in ((a, b, False), (b, a, True)):
        result = judge(prompt, left, right)
        if not isinstance(result, dict):
            raise ValueError("judge response requires winner A/B/tie and a reason string")
        # External evaluator output has not yet been validated.
        fields = cast(dict[object, object], result)
        winner = fields.get("winner")
        if (
            not isinstance(winner, str)
            or winner not in {"A", "B", "tie"}
            or not isinstance(fields.get("reason"), str)
        ):
            raise ValueError("judge response requires winner A/B/tie and a reason string")
        score = {"A": 1.0, "B": 0.0, "tie": 0.5}[winner]
        scores.append(1 - score if swapped else score)
    return {"a_score": sum(scores) / 2, "consistent": scores[0] == scores[1], "calls": 2}


def bootstrap_interval(
    values: FloatValues, *, confidence: float = 0.95, resamples: object = 2000, seed: int = 0
) -> tuple[float, float]:
    array = _values(values)
    if not 0 < confidence < 1 or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("invalid confidence/resamples")
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(array, size=array.size, replace=True).mean() for _ in range(resamples)]
    )
    low, high = np.quantile(means, [(1 - confidence) / 2, (1 + confidence) / 2])
    return float(low), float(high)


def population_stability(
    reference: FloatValues, current: FloatValues, *, bins: object = 10, smoothing: float = 1e-6
) -> float:
    """PSI with training quantile edges and open-ended tails.

    Constant reference gets one split at its value. PSI signals covariate drift,
    not concept drift or an automatic retraining decision.
    """
    ref, cur = _values(reference), _values(current)
    if not isinstance(bins, int) or bins < 2 or not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("bins >=2 and positive finite smoothing required")
    inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.r_[-np.inf, inner, np.inf]
    p = np.histogram(ref, bins=edges)[0].astype(float) + smoothing
    q = np.histogram(cur, bins=edges)[0].astype(float) + smoothing
    p /= p.sum()
    q /= q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def contamination(
    train: Sequence[Example], evaluation: Sequence[Example], *, n: object = 5
) -> dict[str, float]:
    """Fraction of each eval prompt's distinct n-grams seen in other training IDs.

    This diagnostic cannot detect semantic paraphrases or prove uncontaminated
    pretraining. Same-ID records are excluded; callers still enforce split IDs.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be positive")
    dataset_digest(train)
    dataset_digest(evaluation)

    def grams(text: str) -> set[tuple[str, ...]]:
        tokens = re.findall(r"\w+", text.casefold())
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    index: dict[tuple[str, ...], set[str]] = {}
    for example in train:
        for gram in grams(example.prompt):
            index.setdefault(gram, set()).add(example.id)
    result: dict[str, float] = {}
    for example in evaluation:
        query = grams(example.prompt)
        hits = sum(bool(index.get(gram, set()) - {example.id}) for gram in query)
        result[example.id] = hits / len(query) if query else 0.0
    return result


def trajectory_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not rows:
        raise ValueError("trajectories must not be empty")
    validated: list[Trajectory] = []
    for row in rows:
        success = row.get("success")
        if not isinstance(success, bool):
            raise ValueError("success must be boolean")
        counts: dict[str, int] = {}
        for field in ("tool_calls", "unsafe_attempts"):
            value = row.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
            counts[field] = value
        validated.append(
            Trajectory(
                success=success,
                tool_calls=counts["tool_calls"],
                unsafe_attempts=counts["unsafe_attempts"],
            )
        )
    return {
        "task_success": sum(row["success"] for row in validated) / len(rows),
        "safe_success": sum(row["success"] and not row["unsafe_attempts"] for row in validated)
        / len(rows),
        "mean_tool_calls": sum(row["tool_calls"] for row in validated) / len(rows),
    }


def regression_gate(
    baseline: FloatValues, candidate: FloatValues, *, max_drop: float = 0.02, seed: int = 0
) -> RegressionResult:
    """One paired observation per case, higher is better; conservative lower-CI gate."""
    before, after = _values(baseline), _values(candidate)
    if before.shape != after.shape or not math.isfinite(max_drop) or max_drop < 0:
        raise ValueError("paired scores and a nonnegative tolerance required")
    low, high = bootstrap_interval(after - before, seed=seed)
    return {
        "passed": low >= -max_drop,
        "mean_delta": float((after - before).mean()),
        "delta_interval": (low, high),
    }


@dataclass(frozen=True)
class RunManifest:
    model: str
    dataset: str
    evaluator: str
    seed: int

    @property
    def run_id(self) -> str:
        return sha256(_canonical(asdict(self)).encode()).hexdigest()[:16]


def write_results(
    path: str | Path, manifest: RunManifest, rows: Sequence[Mapping[str, object]]
) -> None:
    """Validate/serialize before atomic replacement; no wall-clock entropy.

    Rows are supplied in evaluation order. Include model revision, prompt rubric
    revision and dataset digest in the manifest rather than mutable aliases.
    """
    lines = [_canonical({"manifest": asdict(manifest), "run_id": manifest.run_id})]
    seen: set[str] = set()
    for row in rows:
        ident = row.get("id")
        if not isinstance(ident, str) or ident in seen:
            raise ValueError("result IDs must be unique strings")
        seen.add(ident)
        lines.append(_canonical(row))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as out:
            temporary = Path(out.name)
            out.write("\n".join(lines) + "\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
