import pytest
import torch
from nano_gpt_ssm import SelectiveSSM, TinySSMLanguageModel, diagonal_ssm_scan


def test_diagonal_recurrence_matches_hand_calculation() -> None:
    inputs = torch.tensor([[[2.0], [4.0], [0.0]]])
    # s_t = .5*s_(t-1) + .25*x_t, y_t = 2*s_t + .1*x_t
    outputs, final_state = diagonal_ssm_scan(
        inputs,
        decay=torch.full((1, 3, 1), 0.5),
        write=torch.full((1, 3, 1), 0.25),
        read=torch.full((1, 3, 1), 2.0),
        skip=torch.tensor([0.1]),
    )
    assert torch.allclose(outputs, torch.tensor([[[1.2], [2.9], [1.25]]]))
    assert torch.allclose(final_state, torch.tensor([[0.625]]))


def test_initial_state_contributes_without_mutation() -> None:
    initial = torch.tensor([[2.0]])
    output, final = diagonal_ssm_scan(
        torch.zeros(1, 2, 1),
        decay=torch.full((1, 2, 1), 0.5),
        write=torch.ones(1, 2, 1),
        read=torch.ones(1, 2, 1),
        skip=torch.zeros(1),
        initial_state=initial,
    )
    assert torch.equal(output, torch.tensor([[[1.0], [0.5]]]))
    assert torch.equal(final, torch.tensor([[0.5]]))
    assert torch.equal(initial, torch.tensor([[2.0]]))


def test_selective_parameters_are_input_dependent_and_decay_is_stable() -> None:
    torch.manual_seed(7)
    layer = SelectiveSSM(4)
    x = torch.randn(2, 6, 4)
    decay, write, read = layer.coefficients(x)
    assert decay.shape == write.shape == read.shape == x.shape
    assert torch.all((decay > 0) & (decay < 1))
    changed = layer.coefficients(x + 1)
    assert all(not torch.equal(a, b) for a, b in zip((decay, write, read), changed, strict=True))


def test_chunked_scan_matches_full_scan() -> None:
    torch.manual_seed(8)
    layer = SelectiveSSM(6)
    x = torch.randn(2, 9, 6)
    full, state = layer(x)
    prefix, prefix_state = layer(x[:, :4])
    suffix, suffix_state = layer(x[:, 4:], initial_state=prefix_state)
    assert torch.allclose(torch.cat((prefix, suffix), dim=1), full, atol=1e-6)
    assert torch.allclose(suffix_state, state, atol=1e-6)


def test_ssm_is_causal() -> None:
    torch.manual_seed(9)
    model = TinySSMLanguageModel(7, embedding_size=8, state_size=12).eval()
    ids = torch.randint(7, (2, 8))
    changed = ids.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % 7
    assert model(ids).shape == (2, 8, 7)
    assert torch.allclose(model(ids)[:, :5], model(changed)[:, :5], atol=1e-6)


def test_ssm_gradients_cross_time_and_are_finite() -> None:
    torch.manual_seed(10)
    layer = SelectiveSSM(3)
    inputs = torch.randn(2, 5, 3, requires_grad=True)
    output, _ = layer(inputs)
    output[:, -1].square().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert torch.count_nonzero(inputs.grad[:, 0]) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


@pytest.mark.parametrize("bad_shape", [(2, 0, 4), (2, 3, 5)])
def test_selective_scan_rejects_empty_or_wrong_width(bad_shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        SelectiveSSM(4)(torch.empty(bad_shape))


def test_scan_rejects_wrong_initial_state() -> None:
    with pytest.raises(ValueError, match="initial_state"):
        SelectiveSSM(4)(torch.ones(2, 3, 4), initial_state=torch.zeros(1, 4))
