# %% [markdown]
# Week 20 — a reproducible release decision, not a leaderboard.
# No provider, model download, experiment tracking service, or benchmark download.

# %%
from pathlib import Path
from tempfile import TemporaryDirectory

from evals_mlops import (
    Example,
    RunManifest,
    contamination,
    dataset_digest,
    exact_match,
    population_stability,
    regression_gate,
    split_dataset,
    write_results,
)

examples = [Example(str(i), f"What is {i} plus one?", str(i + 1)) for i in range(30)]
train, validation, test = split_dataset(examples, seed=20)
manifest = RunManifest("arithmetic-rule-v1", dataset_digest(test), "exact-match-v1", 20)
predictions = [str(int(e.id) + 1) for e in test]
print("Fixture-only exact match:", exact_match(predictions, [e.expected for e in test]))
print("Prompt overlap (shared template is expected):", contamination(train, test, n=3))
print("Covariate shift:", population_stability(list(range(30)), list(range(40, 70))))
print("Paired gate:", regression_gate([0, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 1]))
with TemporaryDirectory() as directory:
    path = Path(directory) / "evaluation.jsonl"
    write_results(
        path,
        manifest,
        [{"id": e.id, "prediction": p} for e, p in zip(test, predictions, strict=True)],
    )
    print(path.read_text())

# %% [markdown]
# Next: connect W13 nano_gpt_ssm or W15a post_training_lab outputs, freeze a new
# held-out split, register a rubric before looking at results, and report failures.
