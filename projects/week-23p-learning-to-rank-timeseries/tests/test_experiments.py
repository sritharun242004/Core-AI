import numpy as np
import pytest
from learning_to_rank_timeseries import estimate_ate, randomize, srm_pvalue


def test_assignment_is_reproducible_order_independent_and_unit_stable():
    units = [f"u{i}" for i in range(200)]
    z = randomize(units, seed=23)
    reverse = randomize(units[::-1], seed=23)[::-1]
    assert np.array_equal(z, reverse)
    assert 60 < z.sum() < 140
    assert not np.array_equal(z, randomize(units, seed=24))
    with pytest.raises(ValueError, match="unique"):
        randomize(["duplicate", "duplicate"])


def test_ate_standard_error_and_confidence_interval_numeric_example():
    result = estimate_ate(np.array([1.0, 3.0, 4.0, 6.0]), np.array([0, 0, 1, 1]))
    assert result.ate == pytest.approx(3)
    assert result.se == pytest.approx(np.sqrt(2))
    assert result.low == pytest.approx(3 - 1.95996398454 * np.sqrt(2))
    assert result.high > result.ate > result.low
    assert (result.n_control, result.n_treated) == (2, 2)


def test_simulated_randomized_effect_interval_and_larger_sample_uncertainty():
    rng = np.random.default_rng(23)
    n = 10000
    z = randomize([str(i) for i in range(n)])
    y = rng.normal(0, 1, n) + 0.4 * z
    full = estimate_ate(y, z, confidence=0.99)
    small = estimate_ate(y[:200], z[:200])
    assert full.low < 0.4 < full.high
    assert full.se < small.se / 4
    assert abs(full.ate - 0.4) < 0.06


def test_confidence_is_long_run_coverage_not_guaranteed_single_run_truth():
    rng = np.random.default_rng(5)
    z = np.tile([0, 1], 500)
    covered = 0
    for _ in range(100):
        report = estimate_ate(rng.normal(size=1000) + 0.4 * z, z, confidence=0.9)
        covered += report.low <= 0.4 <= report.high
    assert 82 <= covered <= 98


def test_srm_flags_bad_allocation_but_not_balanced_counts():
    assert srm_pvalue(500, 500) == 1.0
    assert srm_pvalue(900, 100) < 1e-10
    assert srm_pvalue(800, 200, treatment_probability=0.8) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        srm_pvalue(1, 1)


@pytest.mark.parametrize(
    "outcomes,assignment",
    [
        ([1, 2], [0, 1]),
        ([1, 2, 3, 4], [0, 0, 0, 0]),
        ([1, 2, 3, float("nan")], [0, 0, 1, 1]),
        ([1, 2, 3, 4], [0, 0, 1, 2]),
    ],
)
def test_invalid_or_underpowered_arm_counts_rejected(
    outcomes: list[float | int], assignment: list[int]
):
    with pytest.raises(ValueError):
        estimate_ate(np.array(outcomes), np.array(assignment))
