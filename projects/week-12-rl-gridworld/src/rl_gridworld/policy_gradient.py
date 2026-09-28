"""A small REINFORCE policy-gradient baseline for the same MDP."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .env import Gridworld
from .optimizer import make_optimizer


def softmax_probabilities(logits: Tensor) -> Tensor:
    """Convert logits to normalized action probabilities along the final axis."""

    if logits.ndim == 0:
        raise ValueError("logits must have at least one dimension")
    return torch.softmax(logits, dim=-1)


def discounted_returns(rewards: Iterable[float] | Tensor, gamma: float = 0.99) -> Tensor:
    """Compute ``G_t = r_t + gamma G_(t+1)`` without any environment state."""

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    output = torch.zeros_like(values)
    running = torch.zeros((), dtype=values.dtype, device=values.device)
    for index in range(values.numel() - 1, -1, -1):
        running = values[index] + gamma * running
        output[index] = running
    return output


def policy_gradient_loss(log_probs: Tensor, returns: Tensor) -> Tensor:
    """REINFORCE loss; maximizing return is minimizing ``-log pi(a|s) G``."""

    if log_probs.shape != returns.shape:
        raise ValueError("log_probs and returns must have the same shape")
    return -(log_probs * returns.detach()).mean()


def _one_hot(states: Tensor, n_states: int) -> Tensor:
    state_tensor = torch.as_tensor(states, dtype=torch.long)
    if state_tensor.numel() and (
        int(state_tensor.min()) < 0 or int(state_tensor.max()) >= n_states
    ):
        raise ValueError("state is outside the policy input range")
    return torch.nn.functional.one_hot(state_tensor, num_classes=n_states).float()


class PolicyNetwork(nn.Module):
    """State-to-action logits with one-hot state features.

    One-hot inputs keep the tabular state semantics visible while allowing an
    optional hidden layer to demonstrate a small function-approximator policy.
    """

    def __init__(self, n_states: int, n_actions: int, *, hidden_size: int = 0) -> None:
        super().__init__()
        if n_states <= 0 or n_actions <= 0:
            raise ValueError("n_states and n_actions must be positive")
        if hidden_size < 0:
            raise ValueError("hidden_size cannot be negative")
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.network = (
            nn.Sequential(
                nn.Linear(n_states, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, n_actions),
            )
            if hidden_size
            else nn.Linear(n_states, n_actions)
        )

    def forward(self, states: Tensor | int) -> Tensor:
        return self.network(_one_hot(torch.as_tensor(states), self.n_states))

    def action_probabilities(self, states: Tensor | int) -> Tensor:
        return softmax_probabilities(self(states))

    def distribution(self, states: Tensor | int) -> Categorical:
        return Categorical(probs=self.action_probabilities(states))


def _episode(
    env: Gridworld,
    policy: PolicyNetwork,
    *,
    greedy: bool = False,
) -> tuple[list[Tensor], list[float], bool]:
    state = env.reset()
    log_probs: list[Tensor] = []
    rewards: list[float] = []
    succeeded = False
    for _ in range(env.max_steps):
        distribution = policy.distribution(torch.tensor(state))
        action = distribution.probs.argmax().item() if greedy else distribution.sample().item()
        log_probs.append(distribution.log_prob(torch.tensor(action)))
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            succeeded = bool(info["terminated"])
            break
    return log_probs, rewards, succeeded


def evaluate_policy(
    env: Gridworld,
    policy: PolicyNetwork,
    *,
    episodes: int = 20,
    greedy: bool = True,
) -> dict[str, float]:
    """Evaluate a policy without updating it."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    rewards: list[float] = []
    successes = 0
    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        for _ in range(episodes):
            _, episode_rewards, succeeded = _episode(env, policy, greedy=greedy)
            rewards.append(float(sum(episode_rewards)))
            successes += int(succeeded)
    if was_training:
        policy.train()
    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": successes / episodes,
    }


def train_policy_gradient(
    env: Gridworld,
    *,
    episodes: int = 300,
    learning_rate: float = 0.03,
    gamma: float = 0.98,
    seed: int = 0,
    hidden_size: int = 0,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    entropy_coefficient: float = 0.01,
    evaluation_episodes: int = 20,
) -> dict[str, object]:
    """Train a seeded REINFORCE baseline and return inspectable metrics."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if learning_rate <= 0 or weight_decay < 0 or entropy_coefficient < 0:
        raise ValueError("learning_rate must be positive; penalties cannot be negative")
    torch.manual_seed(seed)
    policy = PolicyNetwork(env.n_states, env.n_actions, hidden_size=hidden_size)
    optimizer = make_optimizer(
        policy.parameters(),
        name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    rewards: list[float] = []
    losses: list[float] = []
    moving_average: list[float] = []
    for episode in range(episodes):
        policy.train()
        log_probs, episode_rewards, _ = _episode(env, policy)
        returns = discounted_returns(episode_rewards, gamma=gamma)
        if returns.numel() > 1 and float(returns.std(unbiased=False)) > 1e-8:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        log_prob_tensor = torch.stack(log_probs)
        loss = policy_gradient_loss(log_prob_tensor, returns)
        if entropy_coefficient:
            # Re-run the state-independent entropy estimate for a light
            # exploration bonus; it is deliberately not hidden in the loss.
            entropy = torch.stack([
                policy.distribution(torch.tensor(state)).entropy()
                for state in range(env.n_states)
            ]).mean()
            loss = loss - entropy_coefficient * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        reward = float(sum(episode_rewards))
        rewards.append(reward)
        losses.append(float(loss.detach()))
        moving_average.append(float(np.mean(rewards[max(0, episode - 19) : episode + 1])))
    evaluation = evaluate_policy(env, policy, episodes=evaluation_episodes, greedy=True)
    return {
        "policy": policy,
        "episode_rewards": np.asarray(rewards, dtype=np.float64),
        "losses": np.asarray(losses, dtype=np.float64),
        "moving_average": np.asarray(moving_average, dtype=np.float64),
        "evaluation_mean_reward": evaluation["mean_reward"],
        "evaluation_success_rate": evaluation["success_rate"],
    }


softmax = softmax_probabilities
reinforce = train_policy_gradient
