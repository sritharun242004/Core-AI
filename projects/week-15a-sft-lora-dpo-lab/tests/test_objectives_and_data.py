import copy

import pytest
import torch
from post_training_lab import (
    TinyCausalLM,
    dpo_batch_loss,
    dpo_loss,
    ipo_loss,
    kto_loss,
    make_preference_pairs,
    make_sft_examples,
    orpo_loss,
    sequence_logprob,
    sft_loss,
    simpo_loss,
)


def test_deterministic_preference_fixture_and_sft_loss_are_finite() -> None:
    first = make_preference_pairs()
    second = make_preference_pairs()
    assert [(p.prompt, p.chosen, p.rejected) for p in first] == [
        (p.prompt, p.chosen, p.rejected) for p in second
    ]
    examples = make_sft_examples()
    model = TinyCausalLM(vocab_size=16, d_model=16, n_heads=4, max_seq_len=8)
    logits = model(torch.stack([example.input_ids for example in examples]))
    loss = sft_loss(logits, torch.stack([example.labels for example in examples]))
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_sequence_logprob_respects_mask() -> None:
    logits = torch.zeros(2, 3, 5)
    labels = torch.tensor([[0, 1, 2], [2, 3, 4]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    values = sequence_logprob(logits, labels, mask=mask)
    assert values.shape == (2,)
    expected = torch.tensor([-2 * torch.log(torch.tensor(5.0)), -torch.log(torch.tensor(5.0))])
    assert torch.allclose(values, expected)


def test_sequence_logprob_ignores_ignore_index_and_can_normalize() -> None:
    logits = torch.zeros(1, 3, 5)
    labels = torch.tensor([[0, -100, 2]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    summed = sequence_logprob(logits, labels, mask=mask)
    averaged = sequence_logprob(logits, labels, mask=mask, normalize=True)
    token_logp = -torch.log(torch.tensor(5.0))
    assert torch.allclose(summed, torch.tensor([2 * token_logp]))
    assert torch.allclose(averaged, torch.tensor([token_logp]))


def test_dpo_batch_loss_uses_causal_next_token_scores_and_freezes_reference_gradients() -> None:
    torch.manual_seed(21)
    policy = TinyCausalLM(vocab_size=8, d_model=8, n_heads=2, max_seq_len=5).eval()
    reference = copy.deepcopy(policy).eval()
    with torch.no_grad():
        reference.token_embedding.weight.add_(0.2)
    chosen = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    rejected = torch.tensor([[1, 2, 5, 4], [2, 3, 1, 5]])
    loss = dpo_batch_loss(policy, reference, chosen, rejected)
    with torch.no_grad():
        reference_chosen = sequence_logprob(reference(chosen)[:, :-1], chosen[:, 1:])
        reference_rejected = sequence_logprob(reference(rejected)[:, :-1], rejected[:, 1:])
    expected = dpo_loss(
        sequence_logprob(policy(chosen)[:, :-1], chosen[:, 1:]),
        sequence_logprob(policy(rejected)[:, :-1], rejected[:, 1:]),
        reference_chosen,
        reference_rejected,
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_kto_uses_logistic_utility_and_nonnegative_detached_baseline() -> None:
    policy = torch.tensor([0.4, -0.2])
    reference = torch.zeros_like(policy)
    desirable = torch.tensor([True, False])
    beta = 0.5
    baseline = (policy - reference).mean().clamp_min(0.0)
    expected = torch.where(
        desirable,
        1 - torch.sigmoid(beta * (policy - reference - baseline)),
        1 - torch.sigmoid(beta * (baseline - policy + reference)),
    ).mean()
    assert torch.allclose(kto_loss(policy, reference, desirable, beta=beta), expected)


def test_orpo_uses_probability_odds_not_a_raw_log_probability_ratio() -> None:
    chosen = torch.log(torch.tensor([0.5]))
    rejected = torch.log(torch.tensor([0.25]))
    chosen_odds = chosen - torch.log1p(-torch.exp(chosen))
    rejected_odds = rejected - torch.log1p(-torch.exp(rejected))
    expected = -torch.nn.functional.logsigmoid(chosen_odds - rejected_odds).mean()
    assert torch.allclose(orpo_loss(chosen, rejected, beta=1.0), expected)


def test_simpo_normalizes_sequence_scores_before_applying_reward_margin() -> None:
    chosen = torch.tensor([-2.0, -4.0])
    rejected = torch.tensor([-3.0, -2.0])
    chosen_lengths = torch.tensor([2, 4])
    rejected_lengths = torch.tensor([3, 2])
    expected = -torch.nn.functional.logsigmoid(
        2.0 * (chosen / chosen_lengths - rejected / rejected_lengths) - 0.5
    ).mean()
    assert torch.allclose(
        simpo_loss(
            chosen,
            rejected,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
        ),
        expected,
    )


@pytest.mark.parametrize("objective", [dpo_loss, ipo_loss])
def test_pair_objectives_detach_reference_and_match_equations(objective) -> None:
    chosen = torch.tensor([-1.0, -2.0], requires_grad=True)
    rejected = torch.tensor([-3.0, -2.5], requires_grad=True)
    ref_chosen = torch.tensor([-2.0, -2.5], requires_grad=True)
    ref_rejected = torch.tensor([-2.5, -3.0], requires_grad=True)
    margin = (chosen - rejected) - (ref_chosen - ref_rejected)
    beta = 0.4
    expected = (
        -torch.nn.functional.logsigmoid(beta * margin).mean()
        if objective is dpo_loss
        else (margin - 1 / (2 * beta)).square().mean()
    )
    loss = objective(chosen, rejected, ref_chosen, ref_rejected, beta=beta)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert chosen.grad is not None and rejected.grad is not None
    assert ref_chosen.grad is None and ref_rejected.grad is None


def test_kto_detaches_reference_and_baseline_with_correct_gradient() -> None:
    policy = torch.tensor([-1.0, -4.0], requires_grad=True)
    reference = torch.tensor([-3.0, -2.0], requires_grad=True)
    desirable = torch.tensor([True, False])
    beta = 0.3
    loss = kto_loss(policy, reference, desirable, beta=beta)
    loss.backward()
    delta = policy.detach() - reference.detach()
    baseline = delta.mean().clamp_min(0.0)
    good_sigmoid = torch.sigmoid(beta * (delta - baseline))
    bad_sigmoid = torch.sigmoid(beta * (baseline - delta))
    expected_grad = (
        torch.where(
            desirable,
            -good_sigmoid * (1 - good_sigmoid),
            bad_sigmoid * (1 - bad_sigmoid),
        )
        * beta
        / 2
    )
    torch.testing.assert_close(policy.grad, expected_grad)
    assert reference.grad is None


@pytest.mark.parametrize("delta", [-3.0, 2.0])
def test_kto_clamps_toy_baseline_and_weights_binary_utilities(delta: float) -> None:
    policy = torch.tensor([delta, delta])
    expected = (
        2 * (1 - torch.sigmoid(torch.tensor(0.2 * (delta - max(delta, 0)))))
        + 3 * (1 - torch.sigmoid(torch.tensor(0.2 * (max(delta, 0) - delta))))
    ) / 2
    torch.testing.assert_close(
        kto_loss(
            policy,
            torch.zeros(2),
            torch.tensor([True, False]),
            beta=0.2,
            desirable_weight=2,
            undesirable_weight=3,
        ),
        expected,
    )


def test_orpo_remains_stable_near_probability_one_and_at_tiny_probabilities() -> None:
    chosen = torch.tensor([-1e-10, -1000.0], dtype=torch.float64, requires_grad=True)
    rejected = torch.tensor([-2e-10, -1001.0], dtype=torch.float64, requires_grad=True)
    anchor = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    loss = orpo_loss(chosen, rejected, anchor, beta=0.3)
    log_odds = chosen - torch.log(-torch.expm1(chosen))
    rejected_odds = rejected - torch.log(-torch.expm1(rejected))
    expected = anchor - 0.3 * torch.nn.functional.logsigmoid(log_odds - rejected_odds).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(chosen.grad).all() and torch.isfinite(rejected.grad).all()
    torch.testing.assert_close(anchor.grad, torch.tensor(1.0, dtype=torch.float64))


@pytest.mark.parametrize("bad_logp", [0.2, float("nan")])
def test_orpo_rejects_undefined_odds(bad_logp: float) -> None:
    with pytest.raises(ValueError, match="non-positive"):
        orpo_loss(torch.tensor([bad_logp]), torch.tensor([-1.0]))


def test_orpo_accepts_saturated_probability_one_logps() -> None:
    loss = orpo_loss(torch.tensor([0.0]), torch.tensor([-1.0]))
    assert torch.isfinite(loss)


@pytest.mark.parametrize("objective", [dpo_loss, ipo_loss, orpo_loss, simpo_loss])
def test_pair_objectives_reject_accidental_broadcasting(objective) -> None:
    with pytest.raises(ValueError, match="shape"):
        objective(torch.ones(2, 1), torch.ones(2))


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
def test_sequence_mask_is_binary_and_ignored_positions_have_zero_gradient(dtype) -> None:
    logits = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, 1.0, 0.0], [2.0, 1.0, 0.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[2, -100, 1]])
    mask = torch.tensor([[1, 1, 0]], dtype=dtype)
    before_labels, before_mask = labels.clone(), mask.clone()
    loss = sequence_logprob(logits, labels, mask).sum()
    torch.testing.assert_close(loss, logits[0, 0].log_softmax(0)[2])
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, 1:]) == 0
    assert torch.equal(labels, before_labels) and torch.equal(mask, before_mask)


def test_sequence_normalization_rejects_empty_completion_but_sum_is_zero() -> None:
    logits = torch.randn(2, 3, 4, requires_grad=True)
    labels = torch.full((2, 3), -100)
    summed = sequence_logprob(logits, labels)
    torch.testing.assert_close(summed, torch.zeros(2))
    summed.sum().backward()
    assert torch.count_nonzero(logits.grad) == 0
    with pytest.raises(ValueError, match="token"):
        sequence_logprob(logits, labels, normalize=True)


def test_dpo_batch_shifts_completion_masks_with_targets() -> None:
    torch.manual_seed(35)
    policy = TinyCausalLM(vocab_size=8, d_model=8, n_heads=2, max_seq_len=5).eval()
    reference = TinyCausalLM(vocab_size=8, d_model=8, n_heads=2, max_seq_len=5).eval()
    chosen = torch.tensor([[1, 2, 3, 4, 0]])
    rejected = torch.tensor([[1, 2, 5, 0]])
    chosen_mask = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.bool)
    rejected_mask = torch.tensor([[0, 0, 1, 0]], dtype=torch.bool)

    def score(model, ids, mask):
        logps = model(ids)[:, :-1].log_softmax(-1)
        tokens = logps.gather(-1, ids[:, 1:, None]).squeeze(-1)
        return tokens.masked_fill(~mask[:, 1:], 0).sum(-1)

    expected = dpo_loss(
        score(policy, chosen, chosen_mask),
        score(policy, rejected, rejected_mask),
        score(reference, chosen, chosen_mask),
        score(reference, rejected, rejected_mask),
    )
    actual = dpo_batch_loss(
        policy,
        reference,
        chosen,
        rejected,
        chosen_mask=chosen_mask,
        rejected_mask=rejected_mask,
    )
    torch.testing.assert_close(actual, expected)


def test_dpo_reference_evaluation_restores_modes_and_does_not_mutate_buffers() -> None:
    class StatefulPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 8)
            self.norm = torch.nn.BatchNorm1d(8)
            self.dropout = torch.nn.Dropout(0.8)

        def forward(self, ids):
            values = self.embedding(ids).transpose(1, 2)
            return self.dropout(self.norm(values).transpose(1, 2))

    policy = StatefulPolicy().eval()
    reference = copy.deepcopy(policy).train()
    reference.dropout.eval()
    before_modes = [module.training for module in reference.modules()]
    before_state = {name: value.clone() for name, value in reference.state_dict().items()}
    chosen, rejected = torch.tensor([[1, 2, 3]]), torch.tensor([[1, 4, 3]])
    first_loss = dpo_batch_loss(policy, reference, chosen, rejected)
    second_loss = dpo_batch_loss(policy, reference, chosen, rejected)
    torch.testing.assert_close(first_loss, second_loss)
    torch.testing.assert_close(first_loss, torch.log(torch.tensor(2.0)))
    assert [module.training for module in reference.modules()] == before_modes
    for name, value in reference.state_dict().items():
        assert torch.equal(value, before_state[name])


@pytest.mark.parametrize("length", [0, 1])
def test_dpo_batch_rejects_sequences_without_next_token_targets(length: int) -> None:
    model = TinyCausalLM(vocab_size=8, d_model=8, n_heads=2, max_seq_len=4)
    ids = torch.zeros(1, length, dtype=torch.long)
    with pytest.raises(ValueError, match="two"):
        dpo_batch_loss(model, copy.deepcopy(model), ids, ids)


def test_sft_all_ignored_positions_have_finite_zero_gradients() -> None:
    logits = torch.full((1, 2, 3), float("-inf"), requires_grad=True)
    loss = sft_loss(logits, torch.full((1, 2), -100))
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_preference_objectives_are_finite_and_prefer_chosen_margin() -> None:
    policy_chosen = torch.tensor([[-1.0, -0.5]])
    policy_rejected = torch.tensor([[-1.2, -0.8]])
    reference_chosen = torch.tensor([[-1.1, -0.6]])
    reference_rejected = torch.tensor([[-1.0, -1.0]])
    chosen = policy_chosen.sum(-1)
    rejected = policy_rejected.sum(-1)
    ref_chosen = reference_chosen.sum(-1)
    ref_rejected = reference_rejected.sum(-1)
    losses = [
        dpo_loss(chosen, rejected, ref_chosen, ref_rejected),
        kto_loss(
            torch.cat([chosen, rejected]),
            torch.cat([ref_chosen, ref_rejected]),
            torch.tensor([True, False]),
        ),
        ipo_loss(chosen, rejected, ref_chosen, ref_rejected),
        orpo_loss(chosen, rejected),
        simpo_loss(chosen, rejected),
    ]
    assert all(torch.isfinite(loss) for loss in losses)
    assert dpo_loss(chosen + 1, rejected - 1, ref_chosen, ref_rejected) < dpo_loss(
        chosen, rejected, ref_chosen, ref_rejected
    )
