"""A tiny deterministic tabular gridworld MDP.

The environment deliberately has no framework or data dependency.  A state is
an integer encoding an ``(x, y)`` cell, actions are up/right/down/left, and a
fixed step penalty makes shorter paths preferable to wandering.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

ACTIONS = ("up", "right", "down", "left")
ACTION_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class Gridworld:
    """Small deterministic episodic MDP with a single terminal goal."""

    def __init__(
        self,
        *,
        width: int = 4,
        height: int = 4,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (3, 3),
        walls: Iterable[tuple[int, int]] = (),
        step_reward: float = -0.01,
        goal_reward: float = 1.0,
        max_steps: int | None = None,
    ) -> None:
        if width < 2 or height < 2:
            raise ValueError("width and height must both be at least 2")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when supplied")
        self.width = width
        self.height = height
        self.start = self._validate_xy(start)
        self.goal = self._validate_xy(goal)
        self.walls = frozenset(self._validate_xy(cell) for cell in walls)
        if self.start in self.walls or self.goal in self.walls:
            raise ValueError("start and goal cannot be walls")
        if self.start == self.goal:
            raise ValueError("start and goal must differ")
        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)
        if not np.isfinite(self.step_reward) or not np.isfinite(self.goal_reward):
            raise ValueError("rewards must be finite")
        self.max_steps = max_steps if max_steps is not None else width * height * 4
        self.n_states = width * height
        self.n_actions = len(ACTIONS)
        self.start_state = self.xy_to_state(self.start)
        self.goal_state = self.xy_to_state(self.goal)
        self.state = self.start_state
        self.steps = 0
        self.done = False

    def _validate_xy(self, xy: Sequence[object]) -> tuple[int, int]:
        if len(xy) != 2:
            raise ValueError("coordinates must contain exactly two integers")
        x, y = xy
        if not isinstance(x, (int, np.integer)) or not isinstance(y, (int, np.integer)):
            raise ValueError("coordinates must be integers")
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise ValueError(f"coordinates must be inside [0, {self.width}) x [0, {self.height})")
        return int(x), int(y)

    def xy_to_state(self, xy: tuple[int, int]) -> int:
        x, y = self._validate_xy(xy)
        return y * self.width + x

    def state_to_xy(self, state: object) -> tuple[int, int]:
        if not isinstance(state, (int, np.integer)) or not 0 <= state < self.n_states:
            raise ValueError(f"state must be an integer in [0, {self.n_states})")
        return int(state) % self.width, int(state) // self.width

    def reset(self) -> int:
        """Start a fresh episode at the fixed start state."""

        self.state = self.start_state
        self.steps = 0
        self.done = False
        return self.state

    def _transition(self, state: int, action: int) -> tuple[int, float, bool, bool]:
        if state == self.goal_state:
            return self.goal_state, 0.0, True, False
        x, y = self.state_to_xy(state)
        dx, dy = ACTION_DELTAS[action]
        candidate = (x + dx, y + dy)
        blocked = (
            not (0 <= candidate[0] < self.width and 0 <= candidate[1] < self.height)
            or candidate in self.walls
        )
        next_xy = (x, y) if blocked else candidate
        next_state = self.xy_to_state(next_xy)
        terminated = next_state == self.goal_state
        reward = self.goal_reward if terminated else self.step_reward
        return next_state, reward, terminated, blocked

    def step(self, action: object) -> tuple[int, float, bool, dict[str, bool]]:
        """Apply one action and return ``(state, reward, done, info)``."""

        if self.done:
            raise RuntimeError("episode is done; call reset() before step()")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.n_actions:
            raise ValueError(f"action must be an integer in [0, {self.n_actions})")
        next_state, reward, terminated, blocked = self._transition(self.state, int(action))
        self.steps += 1
        truncated = self.steps >= self.max_steps and not terminated
        self.state = next_state
        self.done = terminated or truncated
        return (
            next_state,
            reward,
            self.done,
            {
                "terminated": terminated,
                "truncated": truncated,
                "blocked": blocked,
            },
        )

    def transition_table(self) -> np.ndarray:
        """Return ``(state, action, (next_state, reward, terminal))`` dynamics."""

        table = np.empty((self.n_states, self.n_actions, 3), dtype=np.float64)
        for state in range(self.n_states):
            for action in range(self.n_actions):
                next_state, reward, terminated, _ = self._transition(state, action)
                table[state, action] = (next_state, reward, float(terminated))
        return table

    def render(self, state: int | None = None) -> str:
        """Return a compact text rendering useful in a notebook."""

        selected = self.state if state is None else state
        selected_xy = self.state_to_xy(selected)
        rows: list[str] = []
        for y in range(self.height):
            cells: list[str] = []
            for x in range(self.width):
                xy = (x, y)
                marker = (
                    "A"
                    if xy == selected_xy
                    else "G"
                    if xy == self.goal
                    else "#"
                    if xy in self.walls
                    else "."
                )
                cells.append(marker)
            rows.append(" ".join(cells))
        return "\n".join(rows)


# Both spellings are common in introductory RL code; keep one implementation.
GridWorld = Gridworld
