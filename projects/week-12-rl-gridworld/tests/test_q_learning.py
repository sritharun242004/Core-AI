import numpy as np
import pytest
from rl_gridworld import (
    Gridworld,
    TabularQLearner,
    bellman_target,
    evaluate_q_table,
    q_learning_update,
    train_q_learning,
)


def test_bellman_target_handles_terminal_and_bootstrapped_steps() -> None:
    assert bellman_target(1.0, 7.0, gamma=0.9, done=False) == 7.3
    assert bellman_target(1.0, 7.0, gamma=0.9, done=True) == 1.0


def test_q_learning_update_is_the_tabular_bellman_update() -> None:
    updated = q_learning_update(
        current_value=2.0,
        reward=1.0,
        next_max=5.0,
        alpha=0.2,
        gamma=0.9,
        done=False,
    )
    assert updated == 2.0 + 0.2 * (1.0 + 0.9 * 5.0 - 2.0)
    assert q_learning_update(2.0, 1.0, 5.0, alpha=0.2, gamma=0.9, done=True) == 1.8


def test_q_learning_improves_greedy_policy_and_reward() -> None:
    env = Gridworld()
    baseline_reward, baseline_success = evaluate_q_table(
        env, np.zeros((env.n_states, env.n_actions))
    )
    result = train_q_learning(env, episodes=260, alpha=0.35, gamma=0.95, seed=7)
    assert result["q_values"].shape == (env.n_states, env.n_actions)
    assert result["episode_rewards"].shape == (260,)
    assert result["successes"][-1] is True
    # Either right or down starts a shortest route; tie-breaking is not the task.
    assert result["greedy_actions"][env.start_state] in (1, 2)
    assert result["evaluation_mean_reward"] > baseline_reward + 1.0
    assert result["evaluation_success_rate"] > baseline_success
    assert result["evaluation_mean_reward"] >= 0.95 - 1e-8
    assert np.isfinite(result["q_values"]).all()


def test_time_limit_does_not_zero_the_q_bootstrap() -> None:
    env = Gridworld(max_steps=1)
    learner = TabularQLearner(16, 4, alpha=1.0, gamma=0.9, epsilon=0, epsilon_min=0)
    learner.q_values[0, 1] = 0.5
    learner.q_values[1, 1] = 1.0
    learner.fit(env, episodes=1)
    assert learner.q_values[0, 1] == pytest.approx(-0.01 + 0.9)


def test_seeded_q_training_repeats_exactly() -> None:
    first = train_q_learning(Gridworld(), episodes=60, seed=19)
    second = train_q_learning(Gridworld(), episodes=60, seed=19)
    np.testing.assert_array_equal(first["q_values"], second["q_values"])
    np.testing.assert_array_equal(first["episode_rewards"], second["episode_rewards"])


def test_invalid_q_states_and_evaluation_counts_are_rejected() -> None:
    learner = TabularQLearner(16, 4)
    with pytest.raises(ValueError):
        learner.update(-1, 0, 1.0, 1, False)
    with pytest.raises(ValueError):
        evaluate_q_table(Gridworld(), learner.q_values, episodes=0)
