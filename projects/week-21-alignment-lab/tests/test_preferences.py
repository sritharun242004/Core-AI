import copy
import math

import pytest
import torch
from alignment_lab import TinyPolicy, dpo_loss, freeze_reference, preference_fixture, train_dpo


def test_dpo_matches_worked_example_and_detaches_reference():
    chosen = torch.tensor([math.log(0.6)], requires_grad=True)
    rejected = torch.tensor([math.log(0.3)], requires_grad=True)
    ref_chosen = torch.tensor([math.log(0.4)], requires_grad=True)
    ref_rejected = torch.tensor([math.log(0.4)], requires_grad=True)
    loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.5)
    assert loss.item() == pytest.approx(math.log1p(math.exp(-0.5 * math.log(2))))
    loss.backward()
    assert chosen.grad.item() < 0 < rejected.grad.item()
    assert ref_chosen.grad is None and ref_rejected.grad is None


def test_identical_policy_reference_is_log_two_and_extremes_are_stable():
    chosen, rejected = torch.tensor([-1.0, -2.0]), torch.tensor([-3.0, -1.0])
    assert dpo_loss(chosen, rejected, chosen, rejected).item() == pytest.approx(math.log(2))
    loss = dpo_loss(torch.tensor([-1e6]), torch.tensor([0.0]), chosen[:1], rejected[:1])
    assert torch.isfinite(loss)


@pytest.mark.parametrize("beta", [0, -1, float("nan"), float("inf")])
def test_bad_beta_is_rejected(beta):
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(*(torch.zeros(2) for _ in range(4)), beta=beta)


def test_bad_log_probability_inputs_are_rejected():
    with pytest.raises(ValueError):
        dpo_loss(torch.zeros(2), torch.zeros(3), torch.zeros(2), torch.zeros(2))
    with pytest.raises(ValueError):
        dpo_loss(*(torch.empty(0) for _ in range(4)))
    with pytest.raises(ValueError):
        dpo_loss(torch.tensor([float("nan")]), *(torch.zeros(1) for _ in range(3)))


def test_training_really_updates_policy_not_reference_and_improves_preferences():
    policy = TinyPolicy(seed=21)
    reference = freeze_reference(policy)
    policy_before = copy.deepcopy(policy.state_dict())
    reference_before = copy.deepcopy(reference.state_dict())
    pairs = preference_fixture()
    report = train_dpo(policy, reference, pairs, steps=100)
    assert report.initial_loss == pytest.approx(math.log(2), abs=1e-6)
    assert report.final_loss < report.initial_loss * 0.3
    assert report.final_margin > report.initial_margin + 1
    assert report.final_chosen_probability > report.initial_chosen_probability + 0.2
    assert report.final_preference_accuracy == 1
    assert len(report.losses) == 100
    assert any(not torch.equal(v, policy_before[k]) for k, v in policy.state_dict().items())
    assert all(torch.equal(v, reference_before[k]) for k, v in reference.state_dict().items())
    assert not reference.training
    assert all(not p.requires_grad and p.grad is None for p in reference.parameters())


def test_seed_reproducibility_and_initialization_does_not_reseed_caller():
    before = torch.random.get_rng_state().clone()
    first, second = TinyPolicy(seed=42), TinyPolicy(seed=42)
    assert torch.equal(before, torch.random.get_rng_state())
    first_report = train_dpo(first, freeze_reference(first), preference_fixture(), steps=20)
    second_report = train_dpo(second, freeze_reference(second), preference_fixture(), steps=20)
    assert first_report == second_report
    assert all(
        torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters(), strict=True)
    )


def test_mutable_or_shared_reference_is_rejected():
    policy = TinyPolicy()
    with pytest.raises(ValueError, match="frozen"):
        train_dpo(policy, copy.deepcopy(policy), preference_fixture())
    with pytest.raises(ValueError, match="share"):
        train_dpo(policy, policy, preference_fixture())
    reference = freeze_reference(policy)
    reference.embedding.weight = torch.nn.Parameter(
        policy.embedding.weight.detach(), requires_grad=False
    )
    with pytest.raises(ValueError, match="share"):
        train_dpo(policy, reference, preference_fixture())


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"lr": 0}, {"lr": float("nan")}])
def test_training_rejects_invalid_optimizer_settings(kwargs):
    policy = TinyPolicy()
    with pytest.raises(ValueError):
        train_dpo(policy, freeze_reference(policy), preference_fixture(), **kwargs)
