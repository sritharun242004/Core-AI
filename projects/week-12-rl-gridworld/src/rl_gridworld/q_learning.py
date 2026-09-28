"""Tabular Q-learning and Bellman helpers for the gridworld."""

from __future__ import annotations

import numpy as np

from .env import Gridworld


def bellman_target(
    reward: float,
    next_value: float,
    *,
    gamma: float = 0.99,
    done: bool = False,
) -> float:
    """Return ``r + gamma * (1-done) * max_a Q(s', a)``."""

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    return float(reward) if done else float(reward) + gamma * float(next_value)


def q_learning_update(
    current_value: float,
    reward: float,
    next_max: float,
    *,
    alpha: float = 0.1,
    gamma: float = 0.99,
    done: bool = False,
) -> float:
    """Apply one in-place-equivalent tabular Q-learning update."""

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    target = bellman_target(reward, next_max, gamma=gamma, done=done)
    return float(current_value) + alpha * (target - float(current_value))


class TabularQLearner:
    """A seeded epsilon-greedy tabular Q-learning agent."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.985,
        seed: int = 0,
    ) -> None:
        if n_states <= 0 or n_actions <= 0:
            raise ValueError("n_states and n_actions must be positive")
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("alpha must be in (0, 1] and gamma in [0, 1]")
        if not 0 <= epsilon_min <= epsilon <= 1:
            raise ValueError("epsilon_min must be in [0, epsilon] and epsilon <= 1")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_values = np.zeros((n_states, n_actions), dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int, *, greedy: bool = False) -> int:
        if not 0 <= state < self.n_states:
            raise ValueError("state is outside the Q-table")
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        best = np.flatnonzero(self.q_values[state] == self.q_values[state].max())
        return int(self.rng.choice(best)) if not greedy else int(best[0])

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        if not 0 <= state < self.n_states or not 0 <= next_state < self.n_states:
            raise ValueError("state is outside the Q-table")
        if not 0 <= action < self.n_actions:
            raise ValueError("action is outside the Q-table")
        next_max = float(self.q_values[next_state].max())
        new_value = q_learning_update(
            self.q_values[state, action],
            reward,
            next_max,
            alpha=self.alpha,
            gamma=self.gamma,
            done=done,
        )
        self.q_values[state, action] = new_value
        return new_value

    def fit(self, env: Gridworld, episodes: int = 200) -> dict[str, np.ndarray | list[bool]]:
        if env.n_states != self.n_states or env.n_actions != self.n_actions:
            raise ValueError("environment dimensions do not match the Q-table")
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        rewards = np.empty(episodes, dtype=np.float64)
        successes: list[bool] = []
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0.0
            terminated = False
            for _ in range(env.max_steps):
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                self.update(state, action, reward, next_state, info["terminated"])
                state = next_state
                total_reward += reward
                terminated = info["terminated"]
                if done:
                    break
            rewards[episode] = total_reward
            successes.append(bool(terminated))
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return {
            "episode_rewards": rewards,
            "successes": successes,
            "q_values": self.q_values.copy(),
        }

    def greedy_policy(self) -> np.ndarray:
        return self.q_values.argmax(axis=1)


def evaluate_q_table(
    env: Gridworld,
    q_values: np.ndarray,
    *,
    episodes: int = 20,
) -> tuple[float, float]:
    """Evaluate a Q-table greedily, returning mean reward and success rate."""

    if q_values.shape != (env.n_states, env.n_actions):
        raise ValueError("q_values shape does not match environment")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    rewards: list[float] = []
    successes = 0
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        for _ in range(env.max_steps):
            action = int(q_values[state].argmax())
            state, reward, done, info = env.step(action)
            total += reward
            if done:
                successes += int(info["terminated"])
                break
        rewards.append(total)
    return float(np.mean(rewards)), successes / episodes


def train_q_learning(
    env: Gridworld,
    *,
    episodes: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.02,
    epsilon_decay: float = 0.985,
    seed: int = 0,
    evaluation_episodes: int = 20,
) -> dict[str, object]:
    """Train and evaluate a Q-table with a fully deterministic environment."""

    learner = TabularQLearner(
        env.n_states,
        env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    history = learner.fit(env, episodes)
    mean_reward, success_rate = evaluate_q_table(
        env, learner.q_values, episodes=evaluation_episodes
    )
    history.update(
        {
            "greedy_actions": learner.greedy_policy(),
            "evaluation_mean_reward": mean_reward,
            "evaluation_success_rate": success_rate,
            "epsilon": learner.epsilon,
        }
    )
    return history


q_learning = train_q_learning
