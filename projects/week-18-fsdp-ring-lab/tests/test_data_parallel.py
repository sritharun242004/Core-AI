# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportCallIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportIndexIssue=false, reportReturnType=false, reportAssignmentType=false
import pytest
import torch
from fsdp_ring_lab import data_parallel_gradients, full_batch_gradients
from torch import nn


def fixture():
    with torch.random.fork_rng():
        torch.manual_seed(18)
        model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2)).double()
    rng = torch.Generator().manual_seed(19)
    x = torch.randn(7, 3, generator=rng, dtype=torch.float64)
    y = torch.randn(7, 2, generator=rng, dtype=torch.float64)
    return model, x, y


@pytest.mark.parametrize("world", [1, 2, 3, 10])
def test_uneven_data_parallel_matches_full_batch_and_sgd(world):
    model, x, y = fixture()
    expected = full_batch_gradients(model, x, y)
    actual = data_parallel_gradients(model, x, y, world_size=world)
    assert actual.loss == pytest.approx(expected.loss, abs=1e-14)
    for param, got, want in zip(
        model.parameters(), actual.gradients, expected.gradients, strict=True
    ):
        torch.testing.assert_close(got, want, atol=1e-14, rtol=1e-12)
        torch.testing.assert_close(param - 0.1 * got, param - 0.1 * want)
        assert param.grad is None


def test_reproducible_and_does_not_mutate_model_or_rng():
    model, x, y = fixture()
    before = {name: param.clone() for name, param in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    first = data_parallel_gradients(model, x, y, world_size=3)
    second = data_parallel_gradients(model, x, y, world_size=3)
    assert first.loss == second.loss
    for name, param in model.state_dict().items():
        torch.testing.assert_close(param, before[name])
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize(
    "failure", ["empty", "broadcast", "nonfinite", "world", "batchnorm", "dropout"]
)
def test_rejects_unsupported_data_parallel_semantics(failure):
    model, x, y = fixture()
    world = 2
    if failure == "empty":
        x, y = x[:0], y[:0]
    elif failure == "broadcast":
        y = y[:, :1]
    elif failure == "nonfinite":
        y[0, 0] = float("inf")
    elif failure == "world":
        world = 0
    elif failure == "batchnorm":
        model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2)).double()
    else:
        model = nn.Sequential(nn.Dropout(0.2), nn.Linear(3, 2)).double()
    with pytest.raises(ValueError):
        data_parallel_gradients(model, x, y, world_size=world)
