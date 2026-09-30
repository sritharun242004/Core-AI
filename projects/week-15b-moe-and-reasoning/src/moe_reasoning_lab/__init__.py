"""Offline Week 15b lab: sparse MoE routing and verifiable toy reasoning."""

from .moe import ExpertMLP, Routing, TinyMoE, TinyMoEClassifier, TopKRouter
from .reasoning import (
    ArithmeticPolicy,
    arithmetic_reward,
    group_relative_advantages,
    grpo_loss,
    grpo_step,
    majority_vote,
    sample_grouped_answers,
    test_time_majority_vote,
    verify_arithmetic,
)
from .training import make_toy_supervised_data, train_supervised_moe

__all__ = [
    "ArithmeticPolicy",
    "ExpertMLP",
    "Routing",
    "TinyMoE",
    "TinyMoEClassifier",
    "TopKRouter",
    "arithmetic_reward",
    "group_relative_advantages",
    "grpo_loss",
    "grpo_step",
    "majority_vote",
    "make_toy_supervised_data",
    "sample_grouped_answers",
    "test_time_majority_vote",
    "train_supervised_moe",
    "verify_arithmetic",
]
