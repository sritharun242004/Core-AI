"""Week 12: deterministic gridworld, tabular Q-learning, and policy gradients."""

from .env import ACTIONS, GridWorld, Gridworld
from .optimizer import l2_penalty, make_optimizer, regularization_penalty
from .policy_gradient import (
    PolicyNetwork,
    discounted_returns,
    evaluate_policy,
    policy_gradient_loss,
    reinforce,
    softmax,
    softmax_probabilities,
    train_policy_gradient,
)
from .ppo import clipped_surrogate, ppo_clip_demo, ppo_clip_loss, ppo_clip_objective
from .q_learning import (
    TabularQLearner,
    bellman_target,
    evaluate_q_table,
    q_learning,
    q_learning_update,
    train_q_learning,
)

__all__ = [
    "ACTIONS",
    "GridWorld",
    "Gridworld",
    "PolicyNetwork",
    "TabularQLearner",
    "bellman_target",
    "clipped_surrogate",
    "discounted_returns",
    "evaluate_policy",
    "evaluate_q_table",
    "l2_penalty",
    "make_optimizer",
    "policy_gradient_loss",
    "ppo_clip_demo",
    "ppo_clip_loss",
    "ppo_clip_objective",
    "q_learning",
    "q_learning_update",
    "regularization_penalty",
    "reinforce",
    "softmax",
    "softmax_probabilities",
    "train_policy_gradient",
    "train_q_learning",
]
