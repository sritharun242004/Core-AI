# pyright: reportUnknownMemberType=false, reportArgumentType=false, reportAttributeAccessIssue=false
import pytest
import torch
from rl_gridworld import (
    clipped_surrogate,
    make_optimizer,
    ppo_clip_demo,
    ppo_clip_loss,
    regularization_penalty,
)


def test_ppo_clip_keeps_positive_advantage_from_over_updating() -> None:
    ratios = torch.tensor([1.30, 0.70])
    advantages = torch.tensor([2.0, -2.0])
    clipped = clipped_surrogate(ratios, advantages, clip_epsilon=0.2)
    assert torch.allclose(clipped, torch.tensor([2.4, -1.6]))
    assert ppo_clip_loss(ratios, advantages, clip_epsilon=0.2).item() == pytest.approx(-0.4)


def test_ppo_demo_computes_ratio_and_clip_without_a_rollout_dependency() -> None:
    demo = ppo_clip_demo(
        old_log_probs=torch.log(torch.tensor([0.5, 0.5])),
        new_log_probs=torch.log(torch.tensor([0.75, 0.25])),
        advantages=torch.tensor([1.0, -1.0]),
        clip_epsilon=0.2,
    )
    assert torch.allclose(demo["ratios"], torch.tensor([1.5, 0.5]))
    assert demo["objective"].shape == (2,)
    assert demo["clip_epsilon"] == 0.2


def test_supported_optimizers_and_regularization_are_explicit() -> None:
    parameter = torch.nn.Parameter(torch.ones(3))
    for name in ("sgd", "adam", "adamw"):
        optimizer = make_optimizer([parameter], name=name, learning_rate=0.01)
        assert optimizer.param_groups[0]["lr"] == 0.01
    assert regularization_penalty([parameter], coefficient=0.1).item() == pytest.approx(0.3)
