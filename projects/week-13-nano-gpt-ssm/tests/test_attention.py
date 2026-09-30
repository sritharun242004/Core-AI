import math

import pytest
import torch
from nano_gpt_ssm import MLP, CausalSelfAttention, SinusoidalPositionalEncoding


def test_mask_is_lower_triangular_and_includes_self() -> None:
    attention = CausalSelfAttention(8, 2, max_seq_len=5)
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(attention.causal_mask, expected)


def test_probabilities_normalize_over_keys_and_mask_future() -> None:
    torch.manual_seed(2)
    attention = CausalSelfAttention(8, 2, dropout=0.5, max_seq_len=5)
    output, probabilities = attention(torch.randn(3, 5, 8), return_attention=True)
    assert output.shape == (3, 5, 8)
    assert probabilities.shape == (3, 2, 5, 5)
    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(-1), torch.ones(3, 2, 5), atol=1e-6)
    assert torch.count_nonzero(probabilities.triu(diagonal=1)) == 0
    # Returned weights are pre-dropout probabilities, even in training mode.
    assert torch.equal(probabilities[:, :, 0, 0], torch.ones(3, 2))


def test_attention_matches_hand_computed_scaled_dot_product() -> None:
    attention = CausalSelfAttention(2, 1, max_seq_len=2)
    with torch.no_grad():
        attention.query_key_value.weight.copy_(torch.eye(2).repeat(3, 1))
        attention.query_key_value.bias.zero_()
        attention.output.weight.copy_(torch.eye(2))
        attention.output.bias.zero_()
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    output, probabilities = attention(x, return_attention=True)
    second = torch.softmax(torch.tensor([0.0, 1 / math.sqrt(2)]), dim=-1)
    expected = torch.tensor([[[[1.0, 0.0], [second[0], second[1]]]]])
    assert torch.allclose(probabilities, expected, atol=1e-7)
    assert torch.allclose(output, expected[:, 0], atol=1e-7)


def test_future_perturbation_cannot_change_prefix_attention_output() -> None:
    torch.manual_seed(3)
    attention = CausalSelfAttention(12, 3).eval()
    x = torch.randn(2, 7, 12)
    perturbed = x.clone()
    perturbed[:, 4:] = torch.randn_like(perturbed[:, 4:]) * 100
    assert torch.allclose(attention(x)[:, :4], attention(perturbed)[:, :4], atol=1e-6)


def test_position_encoding_matches_formula_and_handles_odd_width() -> None:
    position = SinusoidalPositionalEncoding(5, max_seq_len=3)
    result = position(torch.zeros(2, 3, 5))
    assert result.shape == (2, 3, 5)
    assert torch.equal(result[0, 0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(result[0, 1, :2], torch.tensor([math.sin(1), math.cos(1)]))
    assert torch.allclose(result[0, 1, 2], torch.tensor(math.sin(10_000 ** (-2 / 5))))
    assert not list(position.parameters())


def test_mlp_is_positionwise() -> None:
    torch.manual_seed(4)
    mlp = MLP(8, 24).eval()
    x = torch.randn(2, 5, 8)
    changed = x.clone()
    changed[:, 3:] = 0
    assert mlp(x).shape == x.shape
    assert torch.equal(mlp(x)[:, :3], mlp(changed)[:, :3])


@pytest.mark.parametrize("kwargs", [{"d_model": 7, "n_heads": 2}, {"d_model": 8, "n_heads": 0}])
def test_attention_rejects_incompatible_heads(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        CausalSelfAttention(**kwargs)


def test_attention_rejects_empty_time_axis() -> None:
    with pytest.raises(ValueError, match="time"):
        CausalSelfAttention(8, 2)(torch.empty(2, 0, 8))
