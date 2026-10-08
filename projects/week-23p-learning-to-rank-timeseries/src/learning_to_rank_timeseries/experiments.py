"""Unit-level Bernoulli assignment and unadjusted randomized-experiment inference."""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _valid_units(units: Sequence[object]) -> bool:
    return all(isinstance(unit, str) for unit in units)


def _valid_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def randomize(
    units: list[str], *, treatment_probability: float = 0.5, seed: int = 23
) -> NDArray[np.int64]:
    """Stable SHA256 assignment, independent of input order (not Python hash()).

    A pseudo-random classroom experiment, not a security/eligibility mechanism.
    One row per independent randomization unit, without post-assignment filtering.
    """
    if not 0 < treatment_probability < 1:
        raise ValueError("treatment_probability must be in (0,1)")
    if not _valid_units(units) or len(set(units)) != len(units):
        raise ValueError("units must be unique string IDs")
    assignments: list[bool] = []
    for unit in units:
        digest = hashlib.sha256(f"{seed}:{unit}".encode()).digest()
        uniform = int.from_bytes(digest[:8], "big") / 2**64
        assignments.append(uniform < treatment_probability)
    return np.asarray(assignments, dtype=np.int64)


@dataclass(frozen=True)
class ExperimentResult:
    ate: float
    se: float
    low: float
    high: float
    n_control: int
    n_treated: int
    confidence: float


def estimate_ate(
    outcomes: ArrayLike, assignment: ArrayLike, *, confidence: float = 0.95
) -> ExperimentResult:
    """Difference in unit means, Welch SE and asymptotic normal CI.

    Causal only under random assignment, consistency, no interference, and no
    selective missingness. Not a small-sample exact, sequential or clustered CI.
    """
    y, z = np.asarray(outcomes, dtype=float), np.asarray(assignment)
    if y.ndim != 1 or y.shape != z.shape or not np.isfinite(y).all():
        raise ValueError("finite, aligned one-dimensional observations required")
    if not np.isin(z, [0, 1]).all() or not 0 < confidence < 1:
        raise ValueError("binary assignment and confidence in (0,1) required")
    control, treated = y[z == 0], y[z == 1]
    if min(len(control), len(treated)) < 2:
        raise ValueError("each arm needs at least two independent units")
    ate = float(treated.mean() - control.mean())
    se = float(np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control)))
    critical = NormalDist().inv_cdf((1 + confidence) / 2)
    return ExperimentResult(
        ate, se, ate - critical * se, ate + critical * se, len(control), len(treated), confidence
    )


def srm_pvalue(n_treated: int, n_control: int, *, treatment_probability: float = 0.5) -> float:
    """Two-arm allocation chi-square test (1 degree of freedom), approximate.

    Expected counts must be >=5. A small p-value flags sample ratio mismatch,
    not which logging/randomization bug caused it and not a treatment effect.
    """
    if not all(_valid_count(n) for n in (n_treated, n_control)):
        raise ValueError("counts must be nonnegative integers")
    if not 0 < treatment_probability < 1:
        raise ValueError("treatment probability must be in (0,1)")
    n = n_treated + n_control
    expected_t, expected_c = n * treatment_probability, n * (1 - treatment_probability)
    if min(expected_t, expected_c) < 5:
        raise ValueError("chi-square approximation needs expected counts >=5")
    statistic = (n_treated - expected_t) ** 2 / expected_t + (
        n_control - expected_c
    ) ** 2 / expected_c
    return math.erfc(math.sqrt(statistic / 2))
