# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportCallIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false
import pytest
import torch
from nano_gpt_ssm import (
    CausalSelfAttention,
    ExplicitSSMCell,
    NanoGPT,
    SelectiveSSM,
    diagonal_ssm_scan,
    ssm_recurrence,
)


@pytest.mark.parametrize("max_new_tokens", [0, 1])
def test_generation_validates_full_prompt_before_cropping(max_new_tokens: int) -> None:
    model = NanoGPT(5, d_model=4, n_heads=1, n_layers=1, max_seq_len=2)
    # The invalid id is outside the context that the first generation step reads.
    with pytest.raises(ValueError, match="vocabulary"):
        model.generate(torch.tensor([[5, 1, 2]]), max_new_tokens=max_new_tokens)
    assert model.training


@pytest.mark.parametrize("max_new_tokens", [0, 1])
def test_generation_rejects_empty_prompt_without_changing_mode(max_new_tokens: int) -> None:
    model = NanoGPT(5, d_model=4, n_heads=1, n_layers=1)
    with pytest.raises(ValueError, match="time"):
        model.generate(torch.empty(1, 0, dtype=torch.long), max_new_tokens=max_new_tokens)
    assert model.training


@pytest.mark.parametrize("fail", [False, True])
def test_generation_restores_each_modules_mode_even_after_error(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    model = NanoGPT(5, d_model=4, n_heads=1, n_layers=1)
    model.blocks[0].eval()
    modes = [module.training for module in model.modules()]

    if fail:

        def broken_forward(_):
            raise RuntimeError("injected forward failure")

        monkeypatch.setattr(model, "forward", broken_forward)
        with pytest.raises(RuntimeError, match="injected"):
            model.generate(torch.tensor([[1, 2]]), max_new_tokens=1)
    else:
        model.generate(torch.tensor([[1, 2]]), max_new_tokens=1)
    assert [module.training for module in model.modules()] == modes


def test_generation_matches_manual_greedy_sliding_context() -> None:
    torch.manual_seed(21)
    model = NanoGPT(5, d_model=4, n_heads=1, n_layers=1, max_seq_len=2).eval()
    prompt = torch.tensor([[1, 2, 3], [3, 2, 1]])
    expected = prompt.clone()
    with torch.no_grad():
        for _ in range(3):
            token = model(expected[:, -2:])[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat((expected, token), dim=1)
    actual = model.generate(prompt, max_new_tokens=3)
    assert torch.equal(actual, expected)
    assert not model.training


@pytest.mark.parametrize("state_shape", [(1, 3), (3, 3), (2, 2), (3,)])
def test_explicit_cell_rejects_mismatched_initial_state(state_shape: tuple[int, ...]) -> None:
    cell = ExplicitSSMCell(2, 3)
    with pytest.raises(ValueError, match="initial_state"):
        cell(torch.ones(2, 4, 2), initial_state=torch.zeros(state_shape))


def test_explicit_step_rejects_broadcast_batch() -> None:
    cell = ExplicitSSMCell(2, 3)
    with pytest.raises(ValueError, match="state"):
        cell.step(torch.ones(2, 2), torch.zeros(1, 3))


@pytest.mark.parametrize("scan", ["diagonal", "matrix", "cell", "selective"])
def test_public_recurrences_reject_empty_time(scan: str) -> None:
    inputs = torch.empty(2, 0, 3)
    with pytest.raises(ValueError, match="time"):
        if scan == "diagonal":
            diagonal_ssm_scan(inputs, decay=inputs, write=inputs, read=inputs, skip=torch.zeros(3))
        elif scan == "matrix":
            ssm_recurrence(inputs, torch.eye(3), torch.eye(3))
        elif scan == "cell":
            ExplicitSSMCell(3, 3)(inputs)
        else:
            SelectiveSSM(3).coefficients(inputs)


@pytest.mark.parametrize("kind", ["explicit", "selective"])
def test_chunking_preserves_state_and_gradients_without_mutating_initial_state(kind: str) -> None:
    torch.manual_seed(22)
    cell = ExplicitSSMCell(3, 3) if kind == "explicit" else SelectiveSSM(3)
    inputs = torch.randn(2, 5, 3, requires_grad=True)
    initial = torch.randn(2, 3, requires_grad=True)
    initial_copy = initial.detach().clone()
    full, final = cell(inputs, initial_state=initial)
    full_grads = torch.autograd.grad(full.square().sum() + final.square().sum(), (inputs, initial))
    prefix, state = cell(inputs[:, :2], initial_state=initial)
    suffix, chunk_final = cell(inputs[:, 2:], initial_state=state)
    chunked = torch.cat((prefix, suffix), dim=1)
    chunk_grads = torch.autograd.grad(
        chunked.square().sum() + chunk_final.square().sum(), (inputs, initial)
    )
    assert torch.allclose(full, chunked, atol=1e-6)
    assert torch.allclose(final, chunk_final, atol=1e-6)
    assert torch.equal(initial, initial_copy)
    for full_grad, chunk_grad in zip(full_grads, chunk_grads, strict=True):
        assert torch.allclose(full_grad, chunk_grad, atol=1e-6)
        assert torch.isfinite(chunk_grad).all() and torch.count_nonzero(chunk_grad)


def test_causal_attention_has_no_gradient_path_from_future_inputs() -> None:
    torch.manual_seed(23)
    layer = CausalSelfAttention(4, 2, max_seq_len=5)
    inputs = torch.randn(2, 5, 4, requires_grad=True)
    layer(inputs)[:, :3].square().sum().backward()
    assert torch.equal(inputs.grad[:, 3:], torch.zeros_like(inputs[:, 3:]))
    assert torch.count_nonzero(inputs.grad[:, :3])
