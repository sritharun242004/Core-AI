# rl-gridworld — Week 12

A small, inspectable reinforcement-learning lab. The same deterministic 4×4
Gridworld MDP is solved three ways:

1. **Tabular Q-learning** applies the Bellman optimality update directly.
2. **REINFORCE** is a policy-gradient baseline with a one-hot state policy,
   discounted returns, optional entropy exploration, and an explicit optimizer.
3. **PPO-Clip (conceptual demonstration)** isolates the probability ratio and
   clipped surrogate objective. It is intentionally not presented as a full
   PPO implementation: there is no rollout buffer, critic, GAE, minibatch
   schedule, or claim of PPO benchmark performance.

The project runs on CPU with an in-memory environment. It does not load a
corpus, call a service, or require an accelerator. All tests use deterministic
transitions and seeded action sampling.

## Run the offline checks

From this directory:

```bash
PYTHONPATH=src ../../.venv/bin/python -m pytest
../../.venv/bin/ruff check src tests
```

The percent-format notebook is executable without downloads:

```bash
PYTHONPATH=src ../../.venv/bin/python notebooks/01-rl-gridworld.py
```

Or convert it for Jupyter with a local Jupytext installation:

```bash
PYTHONPATH=src jupytext --to notebook notebooks/01-rl-gridworld.py
```

## Public API

```python
from rl_gridworld import Gridworld, train_policy_gradient, train_q_learning

world = Gridworld()
q_run = train_q_learning(world, episodes=260, seed=7)
pg_run = train_policy_gradient(world, episodes=320, seed=11, hidden_size=16)
print(q_run["evaluation_mean_reward"], pg_run["evaluation_success_rate"])
```

### Environment

`Gridworld` encodes `(x, y)` as `y * width + x`. Actions are ordered
`up=0, right=1, down=2, left=3`. A boundary or wall leaves the agent in place
and returns `step_reward`; entering the goal returns `goal_reward` and ends the
episode. The default is a 4×4 board with start `(0, 0)`, goal `(3, 3)`, a
`-0.01` step cost, and a finite time limit. `transition_table()` exposes the
finite deterministic dynamics for inspection.

### Q-learning

- `bellman_target` makes the terminal bootstrap boundary explicit.
- `q_learning_update` computes `Q ← Q + α [r + γ max Q' − Q]`.
- `TabularQLearner` owns the Q-table and a seeded epsilon-greedy policy.
- `train_q_learning` returns episode rewards, success flags, Q-values, greedy
actions, and a held-out-style greedy evaluation on the same known MDP.

The evaluation is not a generalization score: it reuses this tiny known
transition system. Its purpose is to verify policy improvement and reward.

### Policy gradient

`PolicyNetwork` maps one-hot state features to action logits. The helper
`softmax_probabilities` uses a stable framework softmax, so each state row is a
probability distribution. `discounted_returns` and `policy_gradient_loss`
keep the REINFORCE derivation visible. `train_policy_gradient` supports SGD,
Adam, and AdamW through `make_optimizer`, plus gradient clipping and optional
entropy regularization.

### PPO clip boundary

`ppo_clip_demo` and `clipped_surrogate` implement exactly:

```text
r_t(θ) = exp(log π_θ(a_t|s_t) − log π_old(a_t|s_t))
L_clip = min(r_t A_t, clip(r_t, 1−ε, 1+ε) A_t)
```

Advantages and old log-probabilities are treated as fixed inputs. The demo is
for understanding why a policy update is capped, not an end-to-end PPO claim.

### Optimizers and regularization

`make_optimizer` names SGD, Adam, and AdamW explicitly. `regularization_penalty`
is a coupled L2 penalty that can be added to a minimized loss; AdamW's
`weight_decay` is a decoupled optimizer step. Keeping these paths explicit
prevents silently comparing two kinds of regularization as if they were the
same objective.

## What to inspect

1. Print `Gridworld.transition_table()` and verify that boundary actions are
   self-loops and that the goal row is absorbing.
2. Derive one Q update by hand, then compare it with
   `q_learning_update(current, reward, next_max, alpha, gamma, done)`.
3. Compare a uniform policy's action probabilities with the trained policy's
   probabilities at the start state. Report seed, episode budget, gamma,
   optimizer, and evaluation protocol.
4. Plot the REINFORCE moving average. A single lucky episode is not evidence of
   a reliable policy; reward and success rate need a fixed evaluation policy.
5. Change a PPO ratio from `1.0` to `1.5` with a positive advantage and inspect
   how epsilon `0.2` caps its objective term.

## Honest boundaries

The fixture establishes deterministic MDP transitions, exact Bellman updates,
normalized policy probabilities, finite policy gradients, and reward
improvement on a tiny known task. It does not establish robustness to a new
map, a production control policy, a benchmark comparison, or a complete PPO
trainer. For a larger experiment, specify the map distribution, observation
encoding, rollout/evaluation split, random seeds, and failure budget before
interpreting a return.
