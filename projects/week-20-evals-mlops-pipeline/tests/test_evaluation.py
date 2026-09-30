import json

import numpy as np
import pytest
from evals_mlops import (
    Example,
    RunManifest,
    bootstrap_interval,
    contamination,
    dataset_digest,
    exact_match,
    judge_pair,
    population_stability,
    regression_gate,
    split_dataset,
    trajectory_metrics,
    write_results,
)


def fixture():
    return [Example(str(i), f"question {i}", str(i % 2)) for i in range(20)]


def test_versioning_is_order_independent_but_content_sensitive():
    data = fixture()
    assert dataset_digest(data) == dataset_digest(list(reversed(data)))
    assert dataset_digest(data) != dataset_digest(data[:-1])
    with pytest.raises(ValueError, match="unique"):
        dataset_digest(data + data[:1])


def test_split_is_deterministic_disjoint_and_reorder_invariant():
    data = fixture()
    first = split_dataset(data, seed=7)
    assert first == split_dataset(list(reversed(data)), seed=7)
    assert [len(part) for part in first] == [12, 4, 4]
    assert set.union(*(set(x.id for x in part) for part in first)) == set(x.id for x in data)
    assert not set(x.id for x in first[0]) & set(x.id for x in first[2])


@pytest.mark.parametrize("fractions", [(0.5, 0.2, 0.2), (-0.1, 0.5, 0.6), (1, 0, 0)])
def test_bad_splits_rejected(fractions):
    with pytest.raises(ValueError):
        split_dataset(fixture(), fractions=fractions)


def test_exact_match_normalization_and_length_validation():
    assert exact_match([" YES ", "No"], ["yes", "no"]) == 1
    assert exact_match(["1.0"], ["1"]) == 0
    with pytest.raises(ValueError):
        exact_match([], [])
    with pytest.raises(ValueError):
        exact_match(["a"], [])


def test_judge_counterbalances_position_bias():
    def biased(prompt, a, b):
        return {"winner": "A", "reason": "first"}

    result = judge_pair("q", "good", "bad", biased)
    assert result == {"a_score": 0.5, "consistent": False, "calls": 2}

    def content(prompt, a, b):
        return {"winner": "A" if a == "good" else "B", "reason": "rubric"}

    assert judge_pair("q", "good", "bad", content)["a_score"] == 1
    assert judge_pair("q", "good", "bad", content)["consistent"]


@pytest.mark.parametrize("response", [{"winner": "C"}, {"winner": "A"}, "A"])
def test_invalid_judge_response_is_not_a_silent_pass(response):
    with pytest.raises(ValueError):
        judge_pair("q", "a", "b", lambda *args: response)


def test_bootstrap_seed_and_constant_interval():
    assert bootstrap_interval([1, 1, 1]) == (1, 1)
    assert bootstrap_interval([0, 1, 1, 0], seed=4) == bootstrap_interval([0, 1, 1, 0], seed=4)
    with pytest.raises(ValueError):
        bootstrap_interval([])
    with pytest.raises(ValueError):
        bootstrap_interval([float("nan")])


def test_drift_fixed_training_bins_detects_out_of_range_shift():
    train = np.arange(100, dtype=float)
    assert population_stability(train, train) == pytest.approx(0)
    assert population_stability(train, train + 200) > 1
    assert np.isfinite(population_stability([1, 1, 1], [2, 2, 2]))
    with pytest.raises(ValueError):
        population_stability([1, 2], [float("inf")])


def test_contamination_measures_eval_overlap_and_excludes_same_id():
    train = [Example("a", "One two three four", "x")]
    test = [Example("a", "One two three four", "x"), Example("b", "one two three five", "y")]
    result = contamination(train, test, n=3)
    assert result == {"a": 0.0, "b": 0.5}
    assert contamination(train, [Example("c", "short", "z")], n=3) == {"c": 0.0}


def test_trajectory_metrics_count_success_and_unsafe_attempts_separately():
    report = trajectory_metrics(
        [
            {"success": True, "tool_calls": 2, "unsafe_attempts": 1},
            {"success": False, "tool_calls": 0, "unsafe_attempts": 0},
        ]
    )
    assert report == {"task_success": 0.5, "safe_success": 0.0, "mean_tool_calls": 1.0}
    with pytest.raises(ValueError):
        trajectory_metrics([{"success": "false", "tool_calls": 0, "unsafe_attempts": 0}])


def test_regression_gate_paired_scores_and_budget():
    assert regression_gate([0, 1, 1], [1, 1, 1], max_drop=0)["passed"]
    assert not regression_gate([1, 1], [0, 0], max_drop=0.1)["passed"]
    with pytest.raises(ValueError):
        regression_gate([1], [1, 1])


def test_manifest_and_results_are_reproducible_and_atomic(tmp_path):
    manifest = RunManifest("tiny-policy-v1", dataset_digest(fixture()), "exact-match-v1", 3)
    path = tmp_path / "results.jsonl"
    rows = [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.0}]
    write_results(path, manifest, rows)
    initial = path.read_bytes()
    write_results(path, manifest, rows)
    assert path.read_bytes() == initial
    assert len([json.loads(line) for line in path.read_text().splitlines()]) == 3
    with pytest.raises(ValueError):
        write_results(path, manifest, [{"id": "a", "score": float("nan")}])
    assert path.read_bytes() == initial
