# %% [markdown]
# Week 12 — optimize a policy in a tiny gridworld
#
# This percent-format notebook is offline by default. It uses an in-memory,
# deterministic MDP so every result has a small, inspectable contract.

# %%
import torch
from rl_gridworld import (
    Gridworld,
    clipped_surrogate,
    discounted_returns,
    ppo_clip_demo,
    train_policy_gradient,
    train_q_learning,
)

# %% [markdown]
# ## 1. Inspect the MDP
#
# State ids use `state = y * width + x`; actions are up, right, down, left.
# Boundary actions are self-loops and the goal is terminal/absorbing.

# %%
env = Gridworld()
print(env.render())
print("start state:", env.start_state, "goal state:", env.goal_state)
print("transition table shape:", env.transition_table().shape)

# %% [markdown]
# ## 2. Bellman backup with tabular Q-learning
#
# The target bootstraps from the next state's best action unless the transition
# enters the goal. This run is a teaching smoke test, not a benchmark.

# %%
q_run = train_q_learning(
    env,
    episodes=260,
    alpha=0.35,
    gamma=0.95,
    epsilon_decay=0.985,
    seed=7,
)
print("last five Q-learning rewards:", q_run["episode_rewards"][-5:])
print("greedy actions:", q_run["greedy_actions"])
print("greedy mean reward:", q_run["evaluation_mean_reward"])
print("greedy success rate:", q_run["evaluation_success_rate"])

# %% [markdown]
# ## 3. REINFORCE policy gradient
#
# The policy emits normalized action probabilities. Discounted returns weight
# selected log-probabilities, and the optimizer changes logits through a
# gradient rather than writing a table entry directly.

# %%
pg_run = train_policy_gradient(
    env,
    episodes=320,
    learning_rate=0.08,
    gamma=0.98,
    hidden_size=16,
    seed=11,
)
policy = pg_run["policy"]
with torch.no_grad():
    start_probabilities = policy.action_probabilities(torch.tensor(env.start_state))
print("start action probabilities:", start_probabilities)
print("last five policy rewards:", pg_run["episode_rewards"][-5:])
print("policy greedy mean reward:", pg_run["evaluation_mean_reward"])
print("policy greedy success rate:", pg_run["evaluation_success_rate"])

# %% [markdown]
# ## 4. Optimizer and regularization boundary
#
# The public trainer accepts `sgd`, `adam`, and `adamw`; the default notebook
# run uses Adam. Coupled L2 in a loss and AdamW decay are intentionally distinct
# choices, so an ablation should record which one was used.

# %%
returns = discounted_returns([0.0, 0.0, 1.0], gamma=0.9)
print("example discounted returns:", returns)

# %% [markdown]
# ## 5. PPO-Clip conceptual demonstration
#
# This is not a complete PPO trainer. It only demonstrates how an old/new
# log-probability ratio and a signed advantage enter the clipped surrogate.

# %%
demo = ppo_clip_demo(
    old_log_probs=torch.log(torch.tensor([0.5, 0.5])),
    new_log_probs=torch.log(torch.tensor([0.75, 0.25])),
    advantages=torch.tensor([1.0, -1.0]),
    clip_epsilon=0.2,
)
print("ratios:", demo["ratios"])
print("unclipped objective:", demo["unclipped_objective"])
print("clipped objective:", demo["objective"])
print("minimized clip loss:", demo["loss"])
print("direct objective check:", clipped_surrogate(demo["ratios"], torch.tensor([1.0, -1.0])))

# %% [markdown]
# ## Reflection
#
# Q-learning and REINFORCE solve the same known MDP through different update
# contracts. Q-learning estimates action values and acts greedily; REINFORCE
# estimates a stochastic policy and assigns credit with returns. PPO-Clip is an
# update objective boundary here, not evidence of a trained PPO implementation.
