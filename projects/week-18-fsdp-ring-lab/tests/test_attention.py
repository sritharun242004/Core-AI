# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportCallIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportIndexIssue=false, reportReturnType=false, reportAssignmentType=false
import pytest
import torch
from fsdp_ring_lab import dense_attention, ring_attention


def fixture(length=7, batch=2, scale=1.0):
    rng = torch.Generator().manual_seed(18)
    return tuple(
        (
            torch.randn(batch, 2, length, dim, generator=rng, dtype=torch.float64) * scale
        ).requires_grad_()
        for dim in (3, 3, 4)
    )


@pytest.mark.parametrize("world", [1, 2, 3, 9])
@pytest.mark.parametrize("causal", [True, False])
def test_forward_and_backward_match_independent_dense(world, causal):
    q, k, v = fixture()
    lengths = torch.tensor([7, 4])
    actual = ring_attention(
        q, k, v, world_size=world, query_block_size=2, lengths=lengths, causal=causal
    )
    expected = dense_attention(q, k, v, lengths=lengths, causal=causal)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
    upstream = torch.linspace(-1, 1, actual.numel(), dtype=q.dtype).reshape_as(actual)
    got_grad = torch.autograd.grad((actual * upstream).sum(), (q, k, v))
    want_grad = torch.autograd.grad((expected * upstream).sum(), (q, k, v))
    for got, want in zip(got_grad, want_grad, strict=True):
        torch.testing.assert_close(got, want, atol=2e-12, rtol=1e-10)
        assert torch.count_nonzero(got[1, :, 4:]) == 0


@pytest.mark.parametrize("world", [1, 4])
def test_extreme_logits_and_empty_rows_are_finite(world):
    q, k, v = fixture(scale=1000)
    lengths = torch.tensor([0, 7])
    got = ring_attention(q, k, v, world_size=world, lengths=lengths)
    expected = dense_attention(q, k, v, lengths=lengths)
    torch.testing.assert_close(got, expected)
    assert torch.count_nonzero(got[0]) == 0
    for gradient in torch.autograd.grad(got.sum(), (q, k, v)):
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient[0]) == 0


def test_all_empty_batch_has_zero_outputs_and_gradients():
    q, k, v = fixture()
    got = ring_attention(q, k, v, world_size=3, lengths=torch.tensor([0, 0]))
    assert torch.count_nonzero(got) == 0
    for gradient in torch.autograd.grad(got.sum(), (q, k, v)):
        assert torch.count_nonzero(gradient) == 0


def test_single_token_and_future_tokens_cannot_influence_prefix():
    q, k, v = fixture(length=1)
    torch.testing.assert_close(ring_attention(q, k, v, world_size=4), v)
    q, k, v = fixture()
    expected = ring_attention(q, k, v, world_size=3)
    changed_k, changed_v = k.detach().clone(), v.detach().clone()
    changed_k[:, :, 4:] += 99
    changed_v[:, :, 4:] -= 99
    got = ring_attention(q, changed_k, changed_v, world_size=3)
    torch.testing.assert_close(got[:, :, :4], expected[:, :, :4])


def test_gradcheck_blockwise_online_recurrence():
    q, k, v = fixture(length=3, batch=1)
    assert torch.autograd.gradcheck(
        lambda q, k, v: ring_attention(q, k, v, world_size=2, query_block_size=1),
        (q, k, v),
        fast_mode=True,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"world_size": 0},
        {"query_block_size": 0},
        {"lengths": torch.tensor([8, 3])},
        {"lengths": torch.tensor([3.0, 2.0])},
        {"lengths": torch.tensor([3])},
    ],
)
def test_invalid_attention_options(kwargs):
    with pytest.raises(ValueError):
        ring_attention(*fixture(), **kwargs)


@pytest.mark.parametrize("error", ["nan", "shape", "dtype", "empty", "overflow"])
def test_invalid_attention_tensors(error):
    q, k, v = (t.detach() for t in fixture())
    if error == "nan":
        q[0, 0, 0, 0] = float("nan")
    elif error == "shape":
        k = k[:, :, :-1]
    elif error == "dtype":
        v = v.float()
    elif error == "empty":
        q, k, v = q[:, :, :0], k[:, :, :0], v[:, :, :0]
    else:
        q.fill_(1e308)
        k.fill_(1e308)
    for implementation in (dense_attention, ring_attention):
        with pytest.raises(ValueError):
            implementation(q, k, v)
