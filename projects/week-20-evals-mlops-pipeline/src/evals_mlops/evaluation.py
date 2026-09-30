"""Small reproducible evaluation building blocks, not a benchmark leaderboard."""

import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


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
    fractions=(0.6, 0.2, 0.2),
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


def _values(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("values must be a nonempty finite vector")
    return array


def exact_match(predictions: Sequence[str], expected: Sequence[str]) -> float:
    if not predictions or len(predictions) != len(expected):
        raise ValueError("prediction and target lengths must match and be nonempty")

    def normalize(s):
        return " ".join(s.casefold().split())

    return sum(
        normalize(a) == normalize(b) for a, b in zip(predictions, expected, strict=True)
    ) / len(expected)


def judge_pair(prompt: str, a: str, b: str, judge: Callable) -> dict:
    """Two rubric-judge calls; swap positions, map scores back to original A.

    A contradictory pair averages to .5 but is NOT marked as an agreed tie.
    Adapters must treat candidate text as untrusted data, not instructions.
    """
    scores = []
    for left, right, swapped in ((a, b, False), (b, a, True)):
        result = judge(prompt, left, right)
        if (
            not isinstance(result, dict)
            or result.get("winner") not in {"A", "B", "tie"}
            or not isinstance(result.get("reason"), str)
        ):
            raise ValueError("judge response requires winner A/B/tie and a reason string")
        score = {"A": 1.0, "B": 0.0, "tie": 0.5}[result["winner"]]
        scores.append(1 - score if swapped else score)
    return {"a_score": sum(scores) / 2, "consistent": scores[0] == scores[1], "calls": 2}


def bootstrap_interval(
    values: Sequence[float], *, confidence: float = 0.95, resamples: int = 2000, seed: int = 0
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
    reference: Sequence[float], current: Sequence[float], *, bins: int = 10, smoothing: float = 1e-6
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
    train: Sequence[Example], evaluation: Sequence[Example], *, n: int = 5
) -> dict[str, float]:
    """Fraction of each eval prompt's distinct n-grams seen in other training IDs.

    This diagnostic cannot detect semantic paraphrases or prove uncontaminated
    pretraining. Same-ID records are excluded; callers still enforce split IDs.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be positive")
    dataset_digest(train)
    dataset_digest(evaluation)

    def grams(text):
        tokens = re.findall(r"\w+", text.casefold())
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    index: dict[tuple, set[str]] = {}
    for example in train:
        for gram in grams(example.prompt):
            index.setdefault(gram, set()).add(example.id)
    result = {}
    for example in evaluation:
        query = grams(example.prompt)
        hits = sum(bool(index.get(gram, set()) - {example.id}) for gram in query)
        result[example.id] = hits / len(query) if query else 0.0
    return result


def trajectory_metrics(rows: Sequence[dict]) -> dict[str, float]:
    if not rows:
        raise ValueError("trajectories must not be empty")
    for row in rows:
        if not isinstance(row.get("success"), bool):
            raise ValueError("success must be boolean")
        for field in ("tool_calls", "unsafe_attempts"):
            value = row.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
    return {
        "task_success": sum(row["success"] for row in rows) / len(rows),
        "safe_success": sum(row["success"] and not row["unsafe_attempts"] for row in rows)
        / len(rows),
        "mean_tool_calls": sum(row["tool_calls"] for row in rows) / len(rows),
    }


def regression_gate(
    baseline: Sequence[float], candidate: Sequence[float], *, max_drop: float = 0.02, seed: int = 0
) -> dict:
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


def write_results(path: str | Path, manifest: RunManifest, rows: Sequence[dict]) -> None:
    """Validate/serialize before atomic replacement; no wall-clock entropy.

    Rows are supplied in evaluation order. Include model revision, prompt rubric
    revision and dataset digest in the manifest rather than mutable aliases.
    """
    lines = [_canonical({"manifest": asdict(manifest), "run_id": manifest.run_id})]
    seen = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or row["id"] in seen:
            raise ValueError("result IDs must be unique strings")
        seen.add(row["id"])
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
