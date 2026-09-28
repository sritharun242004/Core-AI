import torch
from rl_gridworld import (
    Gridworld,
    PolicyNetwork,
    discounted_returns,
    policy_gradient_loss,
    softmax_probabilities,
    train_policy_gradient,
)


def test_softmax_probabilities_are_normalized_and_stable() -> None:
    logits = torch.tensor([[1000.0, 999.0, -1000.0], [0.0, 0.0, 0.0]])
    probabilities = softmax_probabilities(logits)
    assert torch.isfinite(probabilities).all()
    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))


def test_discounted_returns_and_reinforce_loss_are_differentiable() -> None:
    returns = discounted_returns([1.0, 0.0, 2.0], gamma=0.5)
    assert torch.allclose(returns, torch.tensor([1.5, 1.0, 2.0]))
    log_probs = torch.tensor([-0.4, -0.7, -0.2], requires_grad=True)
    loss = policy_gradient_loss(log_probs, returns)
    loss.backward()
    assert torch.isfinite(log_probs.grad).all()


def test_policy_gradient_improves_gridworld_reward() -> None:
    torch.manual_seed(11)
    env = Gridworld()
    result = train_policy_gradient(
        env,
        episodes=320,
        learning_rate=0.08,
        gamma=0.98,
        seed=11,
        hidden_size=16,
    )
    assert len(result["episode_rewards"]) == 320
    assert result["evaluation_mean_reward"] > -0.15
    assert result["evaluation_success_rate"] >= 0.8
    assert torch.isfinite(result["policy"].parameters().__next__()).all()


def test_policy_network_returns_a_distribution_for_each_state() -> None:
    policy = PolicyNetwork(n_states=16, n_actions=4, hidden_size=8)
    probabilities = policy.action_probabilities(torch.arange(16))
    assert probabilities.shape == (16, 4)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(16))
