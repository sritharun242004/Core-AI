# pyright: reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false
import numpy as np
import pytest
import torch
from scaling_dpo_repro import (
    dpo_loss,
    fit_power_law,
    paired_seed_interval,
    run_three_sizes,
    train_preference,
)


def test_known_power_law_recovers_exponent_and_constant():
    sizes = np.array([100, 200, 400, 800])
    losses = 1.5 + 4 * sizes**-0.3
    fit = fit_power_law(sizes, losses, irreducible=1.5)
    assert fit["alpha"] == pytest.approx(0.3)
    assert fit["amplitude"] == pytest.approx(4)
    assert fit["rmse_log"] < 1e-10


@pytest.mark.parametrize(
    "sizes,losses", [([1, 1, 1], [3, 2, 1]), ([1, 2], [2, 1]), ([1, 2, 3], [1, 0, 2])]
)
def test_unidentifiable_or_invalid_fit_is_rejected(sizes, losses):
    with pytest.raises(ValueError):
        fit_power_law(sizes, losses)


def test_confidence_is_over_seed_differences():
    lo, hi = paired_seed_interval([1, 2, 3], [2, 3, 4])
    assert lo == pytest.approx(1)
    assert hi == pytest.approx(1)
    with pytest.raises(ValueError):
        paired_seed_interval([1], [1, 2])


def test_dpo_formula_and_reference_detachment():
    chosen = torch.tensor([-1.0], requires_grad=True)
    rejected = torch.tensor([-2.0], requires_grad=True)
    ref_c = torch.tensor([-1.5], requires_grad=True)
    ref_r = torch.tensor([-1.5], requires_grad=True)
    loss = dpo_loss(chosen, rejected, ref_c, ref_r, beta=0.5)
    torch.testing.assert_close(loss, -torch.nn.functional.logsigmoid(torch.tensor(0.5)))
    loss.backward()
    assert chosen.grad.item() < 0 < rejected.grad.item()
    assert ref_c.grad is None and ref_r.grad is None


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_dpo_rejects_nonfinite_scores(index, bad):
    scores = [torch.tensor([-1.0]) for _ in range(4)]
    scores[index] = torch.tensor([bad])
    with pytest.raises(ValueError, match="finite"):
        dpo_loss(*scores)


@pytest.mark.parametrize("runner", [train_preference, run_three_sizes])
def test_cpu_helpers_do_not_reseed_accelerators(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    monkeypatch.setattr(torch.mps, "manual_seed", lambda seed: calls.append(seed))
    before = torch.random.get_rng_state().clone()
    runner(steps=1)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not calls


def test_preference_run_improves_and_is_seeded():
    result = train_preference(seed=3)
    assert result == train_preference(seed=3)
    assert result["after"] > result["before"]
    assert result["reference_unchanged"]


def test_three_sizes_are_real_trained_models_with_heldout_measurements():
    rows = run_three_sizes(seed=2, steps=4)
    assert len(rows) == 3
    assert len({row["parameters"] for row in rows}) == 3
    assert all(row["train_tokens"] > 0 and np.isfinite(row["validation_loss"]) for row in rows)
    assert rows == run_three_sizes(seed=2, steps=4)
