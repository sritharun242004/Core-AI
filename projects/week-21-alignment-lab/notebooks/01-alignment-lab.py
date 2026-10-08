# %% [markdown]
# # Week 21 — separate optimization, behavior, and representation evidence
# Offline CPU only. This is a real DPO update on a tiny categorical policy,
# a SEPARATE rule-based safety fixture, and synthetic interpretability exercises.
# No pretrained model, human preference data, reward model, PPO, or causal claim.

# %%
from dataclasses import asdict
from typing import cast

import torch
import torch.nn.functional as functional
from alignment_lab import (
    Response,
    TinyPolicy,
    activation_metrics,
    evaluate_red_team,
    fit_linear_probe,
    fixture_assistant,
    freeze_reference,
    logit_lens,
    preference_fixture,
    probe_accuracy,
    probe_fixture,
    red_team_fixture,
    sae_fixture,
    train_dpo,
    train_sae,
)

# Small tensor operations benefit from one CPU thread. This script owns its process.
torch.set_num_threads(1)
seed = 21
policy = TinyPolicy(seed=seed)
reference = freeze_reference(policy)
reference_before = {key: value.clone() for key, value in reference.state_dict().items()}
report = train_dpo(policy, reference, preference_fixture())
print("DPO (training pairs only):", {k: v for k, v in asdict(report).items() if k != "losses"})
assert report.final_loss < report.initial_loss
assert report.final_chosen_probability > report.initial_chosen_probability
assert all(
    torch.equal(value, reference_before[key]) for key, value in reference.state_dict().items()
)
assert all(parameter.grad is None for parameter in reference.parameters())

# %% [markdown]
# Each completion is ONE response token. Its log-probability is already the
# completion sum. With a causal LM, sum only shifted completion targets, excluding
# the prompt and padding as in Week 15a. Freeze reference parameters AND buffers;
# no-grad alone would not stop BatchNorm updates in a model with such buffers.

# %%
for case in red_team_fixture():
    print(case.id, case.category, "REFUSE" if case.should_refuse else "ALLOW", case.prompt)

baseline = evaluate_red_team(fixture_assistant)
always_refuse = evaluate_red_team(lambda _prompt, _tools: Response("No.", refused=True))
for name, evaluation in [("parser baseline", baseline), ("always refuse", always_refuse)]:
    print(name, {k: v for k, v in asdict(evaluation).items() if k != "results"})
assert baseline.task_success_rate == 1
assert always_refuse.overrefusal_rate == 1
assert always_refuse.task_success_rate == 0

# %% [markdown]
# The parser has perfect coverage of this deliberately simple grammar. That
# says nothing about the DPO policy or natural-language safety. Tool allowlists
# are enforced outside the assistant. These Python callbacks are trusted code,
# not a sandbox; untrusted real agents need process/container isolation too.

# %%
with torch.no_grad():
    states = policy.hidden_states(torch.arange(8))
    logits = logit_lens(states, policy.final_norm, policy.unembedding)
    assert torch.allclose(logits[:, -1], policy(torch.arange(8)))
    probs = cast(list[float], logits[0].softmax(-1).tolist())  # pyright: ignore[reportUnknownMemberType]
    print("Logit-lens action probabilities, context 0:", probs)

# %% [markdown]
# Intermediate logits use the final LayerNorm and unembedding. An apparent
# action preference in an intermediate state is a diagnostic, not an intention.
# Next use INDEPENDENT synthetic activation datasets, not the eight policy IDs.

# %%
train_x, train_y, test_x, test_y = probe_fixture(seed=seed)
probe = fit_linear_probe(train_x, train_y, seed=seed)
heldout_accuracy = probe_accuracy(probe, test_x, test_y)
print(
    "Probe train / independent test accuracy:",
    probe_accuracy(probe, train_x, train_y),
    heldout_accuracy,
)
print("Flipped-label negative control:", probe_accuracy(probe, test_x, 1 - test_y))
assert heldout_accuracy > 0.9

# %%
sae_train, sae_test = sae_fixture(seed=seed)
sae = train_sae(sae_train, seed=seed)
with torch.no_grad():
    reconstructed, codes = sae.model(sae_test)
    test_mse = functional.mse_loss(reconstructed, sae_test).item()
    mean_baseline_mse = functional.mse_loss(sae_train.mean(0).expand_as(sae_test), sae_test).item()
print("SAE training MSE before / after:", sae.initial_reconstruction, sae.final_reconstruction)
print("SAE independent test MSE / train-mean baseline:", test_mse, mean_baseline_mse)
print("SAE test activation metrics:", asdict(activation_metrics(codes)))
assert test_mse < mean_baseline_mse * 0.4

# %% [markdown]
# ## What would change your mind?
# - DPO: test fresh contexts/preferences before claiming generalization.
# - Safety: freeze unseen paraphrases and manually inspect false refusals.
# - Probes: split by source/task; a random row split can leak shared templates.
# - SAE: sweep L1 on validation, then evaluate once on a frozen test split;
#   report reconstruction AND activity, dead features and seed variability.
# - Causality: preregister interventions/ablations, matched controls and downstream
#   effects. Correlated, decodable features do not establish a causal circuit.
# Public Anthropic dictionary-learning work motivates this exercise. This tiny
# SAE is not a reproduction of full Circuits, feature steering or circuit tracing.
