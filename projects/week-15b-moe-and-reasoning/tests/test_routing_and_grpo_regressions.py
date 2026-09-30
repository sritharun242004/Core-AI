import pytest
import torch
from moe_reasoning_lab import TinyMoE, TopKRouter, group_relative_advantages, grpo_loss


def test_constant_decimal_rewards_have_zero_advantages_and_policy_gradient() -> None:
    rewards = torch.full((2, 7), 0.3)
    advantages = group_relative_advantages(rewards)
    assert torch.equal(advantages, torch.zeros_like(rewards))
    log_probs = torch.full((2, 7), -1.0, requires_grad=True)
    grpo_loss(log_probs, rewards).backward()
    assert torch.equal(log_probs.grad, torch.zeros_like(log_probs))


def test_constant_group_stays_zero_when_batch_also_has_nonconstant_group() -> None:
    rewards = torch.tensor(
        [[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]
    )
    advantages = group_relative_advantages(rewards)
    assert torch.equal(advantages[0], torch.zeros(7))
    assert torch.allclose(advantages[1].mean(), torch.tensor(0.0), atol=1e-6)


def test_advantages_preserve_float64_reward_differences() -> None:
    rewards = torch.tensor([[1e10, 1e10 + 1], [-1e10, -1e10 + 2]], dtype=torch.float64)
    advantages = group_relative_advantages(rewards)
    assert advantages.dtype == torch.float64
    assert torch.allclose(advantages, torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]).double())


@pytest.mark.parametrize("eps", [float("nan"), float("inf")])
def test_advantages_reject_nonfinite_epsilon(eps: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        group_relative_advantages(torch.ones(1, 2), eps=eps)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf")])
def test_advantages_reject_nonfinite_rewards(reward: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        group_relative_advantages(torch.tensor([[0.0, reward]]))


def test_grpo_rejects_empty_prompt_batch() -> None:
    with pytest.raises(ValueError, match="prompts"):
        grpo_loss(torch.empty(0, 2), torch.empty(0, 2))


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_reference_penalty_rejects_invalid_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="kl_weight"):
        grpo_loss(
            torch.zeros(1, 2), torch.ones(1, 2),
            reference_log_probs=torch.zeros(1, 2), kl_weight=weight,
        )


def test_positive_reference_weight_requires_reference() -> None:
    with pytest.raises(ValueError, match="reference_log_probs"):
        grpo_loss(torch.zeros(1, 2), torch.ones(1, 2), kl_weight=0.1)


def test_grpo_clipping_has_correct_gradient_for_each_advantage_sign() -> None:
    ratios = torch.tensor([[0.5, 1.5]])
    current = (ratios.log() - 2).requires_grad_()
    old = torch.full((1, 2), -2.0, requires_grad=True)
    rewards = torch.tensor([[1.0, 0.0]], requires_grad=True)
    grpo_loss(current, rewards, old_log_probs=old).backward()
    assert torch.allclose(current.grad, torch.tensor([[-0.25, 0.75]]))
    assert old.grad is None and rewards.grad is None


def test_reference_penalty_gradient_only_reaches_current_policy() -> None:
    current = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    reference = torch.tensor([[-2.0, -1.0]], requires_grad=True)
    old = current.detach().clone().requires_grad_()
    rewards = torch.ones(1, 2, requires_grad=True)
    grpo_loss(current, rewards, old, reference, kl_weight=0.3).backward()
    assert torch.allclose(current.grad, torch.tensor([[0.3, -0.3]]))
    assert reference.grad is None and old.grad is None and rewards.grad is None


def test_top_one_keeps_selected_probability_so_task_loss_trains_router() -> None:
    model = TinyMoE(d_model=2, hidden_dim=3, num_experts=2, top_k=1, capacity_factor=2)
    with torch.no_grad():
        model.router.gate.weight.copy_(torch.tensor([[0.5, 0.0], [0.0, 0.0]]))
        for expert in model.experts:
            expert.net[2].weight.zero_()
            expert.net[2].bias.fill_(1.0)
    before = model.router.gate.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    output = model(torch.ones(1, 3, 2))
    routing = model.last_routing
    expected_weights = routing.probabilities.gather(1, routing.topk_indices)
    assert torch.allclose(routing.topk_weights, expected_weights)
    output.sum().backward()  # Deliberately no auxiliary loss.
    gradient = model.router.gate.weight.grad
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
    optimizer.step()
    assert not torch.equal(before, model.router.gate.weight)


@pytest.mark.parametrize("dtype,tokens", [(torch.bfloat16, 513), (torch.float16, 65536)])
def test_router_load_counts_remain_exact_in_low_precision(dtype: torch.dtype, tokens: int) -> None:
    router = TopKRouter(1, 2, top_k=1, capacity_factor=0.01).to(dtype=dtype)
    with torch.no_grad():
        router.gate.weight.copy_(torch.tensor([[0.5], [-0.5]], dtype=dtype))
    routing = router(torch.ones(tokens, 1, dtype=dtype))
    assert routing.selected_loads.tolist() == [tokens, 0]
    assert routing.expert_loads.tolist() == [routing.capacity, 0]
    assert torch.isfinite(routing.aux_loss)
    routing.aux_loss.backward()
    gradient = router.gate.weight.grad
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)


def test_balance_gradient_penalizes_requested_load_even_when_dropped() -> None:
    router = TopKRouter(1, 2, top_k=1, capacity_factor=0.1)
    with torch.no_grad():
        router.gate.weight.copy_(torch.tensor([[0.5], [-0.5]]))
    routing = router(torch.ones(10, 1))
    routing.aux_loss.backward()
    probability = torch.sigmoid(torch.tensor(1.0))
    expected = 2 * probability * (1 - probability)
    assert routing.selected_loads.tolist() == [10, 0]
    assert routing.expert_loads.tolist() == [1, 0]
    assert torch.allclose(router.gate.weight.grad, torch.stack((expected, -expected))[:, None])


@pytest.mark.parametrize("factor", [float("nan"), float("inf")])
def test_router_rejects_nonfinite_capacity(factor: float) -> None:
    with pytest.raises(ValueError, match="capacity_factor"):
        TopKRouter(2, 2, capacity_factor=factor)


def test_dropped_tokens_have_no_task_gradient_but_keep_balance_signal() -> None:
    torch.manual_seed(24)
    model = TinyMoE(d_model=2, num_experts=2, top_k=2, capacity_factor=0.1)
    inputs = torch.ones(1, 10, 2, requires_grad=True)
    with torch.no_grad():
        model.router.gate.weight.copy_(torch.tensor([[0.5, 0.0], [0.0, 0.0]]))
    output, aux = model(inputs, return_aux=True)
    output.sum().backward(retain_graph=True)
    assert torch.equal(inputs.grad[:, 1:], torch.zeros_like(inputs[:, 1:]))
    assert torch.count_nonzero(model.router.gate.weight.grad)
    assert all(torch.count_nonzero(expert.net[2].weight.grad) for expert in model.experts)
    assert aux.requires_grad
