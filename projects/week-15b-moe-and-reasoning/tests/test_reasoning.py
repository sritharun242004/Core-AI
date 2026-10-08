# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportCallIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportIndexIssue=false, reportReturnType=false, reportAssignmentType=false
import pytest
import torch
from moe_reasoning_lab.reasoning import (
    ArithmeticPolicy,
    arithmetic_reward,
    group_relative_advantages,
    grpo_loss,
    grpo_step,
    majority_vote,
    sample_grouped_answers,
    verify_arithmetic,
)
from moe_reasoning_lab.reasoning import test_time_majority_vote as run_majority_vote


def test_verifier_accepts_correct_final_answer_and_rejects_wrong_answer() -> None:
    question = "What is 17 + 25?"
    assert verify_arithmetic(question, "17 + 25 = 42")
    assert arithmetic_reward(question, "The answer is 42") == 1.0
    assert not verify_arithmetic(question, "The answer is 41")
    assert arithmetic_reward(question, "41") == 0.0


def test_group_relative_advantages_have_zero_mean_per_prompt() -> None:
    rewards = torch.tensor([[0.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    advantages = group_relative_advantages(rewards)
    assert advantages.shape == rewards.shape
    assert torch.allclose(advantages.mean(dim=-1), torch.zeros(2), atol=1e-7)
    assert torch.equal(advantages[1], torch.zeros(3))


def test_grpo_loss_is_finite_and_policy_has_expected_batch_shape() -> None:
    torch.manual_seed(4)
    policy = ArithmeticPolicy(answer_vocab_size=64)
    prompts = torch.tensor([[2, 3, 0], [5, 4, 1]])
    actions = torch.tensor([[5, 6, 7], [9, 8, 10]])
    rewards = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    logits = policy(prompts).unsqueeze(1).expand(-1, 3, -1)
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    loss = grpo_loss(log_probs, rewards)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_majority_vote_is_deterministic_and_tie_breaks_by_first_seen() -> None:
    assert majority_vote(["42", "41", "42", "41"]) == "42"
    assert majority_vote([3, 2, 2, 3]) == 3
    with pytest.raises(ValueError, match="sample"):
        majority_vote([])


@pytest.mark.parametrize(
    ("question", "response", "expected"),
    [
        ("What is -3 + 2?", "-1", True),
        ("What is 9 - 12?", "-3", True),
        ("What is 2.5 * 4?", "10.0", True),
        ("What is 3 / 2?", "1.5", True),
        ("What is 1 / 0?", "0", False),
        ("What is 2 + 3 * 4?", "5", False),
        ("Run a command; 2 + 2", "4", False),
        ("What is 2 + 2?", "no answer", False),
        ("What is 2 + 2?", "4e10", False),
        ("What is 2 + 2?", "4/1", False),
        ("What is 2 + 2?", "4 or 5", False),
        ("What is 2 + 2?", float("inf"), False),
        ("What is 2 + 2?", "4abc", False),
        ("What is 2 + 2?", "code(4)", False),
        ("What is 2 + 2?", "2 + 2 = 4", True),
        ("What is 2 + 2?", "The answer is 4", True),
    ],
)
def test_verifier_narrow_grammar(question: str, response: str | float, expected: bool) -> None:
    assert verify_arithmetic(question, response) is expected


def test_group_advantages_integer_rewards_and_singleton_are_finite() -> None:
    assert torch.isfinite(group_relative_advantages(torch.tensor([[0, 1, 0]]))).all()
    assert torch.equal(group_relative_advantages(torch.tensor([[1.0]])), torch.zeros(1, 1))


def test_ratio_loss_improves_and_old_policy_is_detached() -> None:
    current = torch.tensor([[-0.8, -1.2]], requires_grad=True)
    old = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    rewards = torch.tensor([[1.0, 0.0]])
    baseline = grpo_loss(old.detach(), rewards, old.detach())
    loss = grpo_loss(current, rewards, old)
    assert loss < baseline
    loss.backward()
    assert old.grad is None
    assert torch.isfinite(current.grad).all()


def test_grouped_update_raises_probability_of_verified_answer() -> None:
    torch.manual_seed(11)
    policy = ArithmeticPolicy(answer_vocab_size=8)
    prompts = torch.tensor([[1, 1, 0]])
    actions = torch.tensor([[2, 3, 2, 4]])
    rewards = torch.tensor([[arithmetic_reward("What is 1 + 1?", int(a)) for a in actions[0]]])
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    with torch.no_grad():
        old = policy(prompts).log_softmax(-1).gather(1, actions)
        before = policy(prompts).softmax(-1)[0, 2].item()
    grpo_step(policy, optimizer, prompts, actions, rewards, old)
    assert policy(prompts).softmax(-1)[0, 2].item() > before
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())


def test_seeded_sampling_and_actual_on_policy_training_reduces_target_nll() -> None:
    torch.manual_seed(5)
    policy = ArithmeticPolicy(answer_vocab_size=8, hidden_dim=16)
    prompts = torch.tensor([[1, 1, 0], [1, 2, 0], [2, 2, 0], [2, 3, 0]])
    questions = ["What is 1 + 1?", "What is 1 + 2?", "What is 2 + 2?", "What is 2 + 3?"]
    targets = torch.tensor([[2], [3], [4], [5]])
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.03)
    generator = torch.Generator().manual_seed(5)
    before = -policy(prompts).log_softmax(-1).gather(1, targets).mean().item()
    for _ in range(45):
        actions, old = sample_grouped_answers(policy, prompts, group_size=32, generator=generator)
        rewards = torch.tensor(
            [
                [arithmetic_reward(q, int(a)) for a in group]
                for q, group in zip(questions, actions, strict=True)
            ]
        )
        grpo_step(policy, optimizer, prompts, actions, rewards, old)
    after = -policy(prompts).log_softmax(-1).gather(1, targets).mean().item()
    assert after < before * 0.5, (before, after)
    first, _ = sample_grouped_answers(policy, prompts, generator=torch.Generator().manual_seed(17))
    second, _ = sample_grouped_answers(policy, prompts, generator=torch.Generator().manual_seed(17))
    assert torch.equal(first, second)


def test_test_time_sampling_is_repeatable_and_counts_calls() -> None:
    calls = []

    def sample(question, rng):
        calls.append(question)
        return rng.choice([4, 4, 3])

    first = run_majority_vote("What is 2 + 2?", sample, num_samples=7, seed=2)
    second = run_majority_vote("What is 2 + 2?", sample, num_samples=7, seed=2)
    assert first == second
    assert len(calls) == 14
    assert first[0] == majority_vote(first[1])


def test_constant_group_has_zero_policy_gradient_and_reference_is_frozen() -> None:
    current = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    reference = torch.tensor([[-2.0, -1.0]], requires_grad=True)
    rewards = torch.ones(1, 2)
    loss = grpo_loss(current, rewards)
    loss.backward()
    assert torch.equal(current.grad, torch.zeros_like(current))
    current.grad = None
    regularized = grpo_loss(current, rewards, reference_log_probs=reference, kl_weight=0.1)
    regularized.backward()
    assert regularized > 0
    assert reference.grad is None


@pytest.mark.parametrize("eps", [0.0, -1.0])
def test_advantages_reject_invalid_epsilon(eps: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        group_relative_advantages(torch.ones(1, 2), eps=eps)
