import numpy as np
import pytest
from rl_gridworld import ACTIONS, Gridworld


def test_gridworld_is_deterministic_and_reaches_goal() -> None:
    first = Gridworld()
    second = Gridworld()
    assert first.reset() == second.reset() == 0
    actions = [1, 1, 1, 2, 2, 2]
    first_trace = [first.step(action) for action in actions]
    second_trace = [second.step(action) for action in actions]
    assert first_trace == second_trace
    assert first_trace[-1][0] == first.goal_state
    assert first_trace[-1][1] == first.goal_reward
    assert first_trace[-1][2] is True


def test_gridworld_boundaries_and_walls_are_stable() -> None:
    env = Gridworld(walls={(1, 0)})
    env.reset()
    state, reward, done, info = env.step(1)  # right is blocked by the wall
    assert state == env.start_state
    assert reward == env.step_reward
    assert done is False
    assert info["blocked"] is True
    assert env.state_to_xy(state) == (0, 0)


def test_gridworld_rejects_invalid_actions_and_encodings() -> None:
    env = Gridworld()
    with pytest.raises(ValueError):
        env.step(len(ACTIONS))
    with pytest.raises(ValueError):
        env.xy_to_state((env.width, 0))
    with pytest.raises(ValueError):
        env.state_to_xy(env.n_states)


def test_gridworld_transition_table_is_local_and_repeatable() -> None:
    env = Gridworld()
    table_a = env.transition_table()
    table_b = env.transition_table()
    assert np.array_equal(table_a, table_b)
    assert table_a.shape == (env.n_states, env.n_actions, 3)
    assert table_a[env.start_state, 0, 0] == env.start_state  # up at top boundary


def test_terminal_model_is_absorbing_with_no_repeated_goal_reward() -> None:
    env = Gridworld()
    terminal_rows = env.transition_table()[env.goal_state]
    assert np.all(terminal_rows[:, 0] == env.goal_state)
    assert np.all(terminal_rows[:, 1] == 0)
    assert np.all(terminal_rows[:, 2] == 1)


def test_goal_termination_and_time_limit_are_distinct() -> None:
    env = Gridworld(width=2, height=2, goal=(1, 1), max_steps=2)
    env.step(1)
    _, reward, done, info = env.step(2)
    assert reward == 1.0 and done
    assert info["terminated"] and not info["truncated"]
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset()
    env.step(0)
    _, _, done, info = env.step(0)
    assert done and info["truncated"] and not info["terminated"]


def test_dynamics_table_does_not_mutate_episode_state() -> None:
    env = Gridworld()
    env.step(1)
    env.transition_table()
    assert env.state == 1 and env.steps == 1 and not env.done


def test_invalid_environment_settings_are_rejected() -> None:
    for kwargs in ({"width": 0}, {"walls": {(0, 0)}}, {"max_steps": 0}, {"step_reward": np.nan}):
        with pytest.raises(ValueError):
            Gridworld(**kwargs)
