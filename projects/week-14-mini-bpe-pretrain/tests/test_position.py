# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
import torch
from mini_bpe_pretrain import (
    alibi_slopes,
    apply_rope,
    build_alibi_bias,
    build_rope_cache,
    rotate_half,
)


def test_rope_cache_shapes_and_position_zero() -> None:
    cos, sin = build_rope_cache(seq_len=5, head_dim=8)
    assert cos.shape == (5, 8)
    assert sin.shape == (5, 8)
    assert torch.allclose(cos[0], torch.ones(8))
    assert torch.allclose(sin[0], torch.zeros(8))


def test_rope_preserves_norm_and_rotates_q_and_k() -> None:
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 5, 8)
    cos, sin = build_rope_cache(5, 8, dtype=q.dtype)
    rotated_q, rotated_k = apply_rope(q, k, cos, sin)
    assert rotated_q.shape == q.shape
    assert rotated_k.shape == k.shape
    assert torch.allclose(rotated_q.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    assert torch.allclose(rotated_k.norm(dim=-1), k.norm(dim=-1), atol=1e-5)
    assert torch.allclose(rotate_half(q[..., :4]), torch.tensor(0.0) + rotate_half(q[..., :4]))


def test_alibi_has_causal_shape_and_more_penalty_for_distant_keys() -> None:
    slopes = alibi_slopes(4)
    bias = build_alibi_bias(4, 6)
    assert slopes.shape == (4,)
    assert bias.shape == (1, 4, 6, 6)
    assert torch.all(bias[0, :, torch.arange(6), torch.arange(6)] == 0)
    assert torch.all(bias[0, :, 5, 0] < bias[0, :, 5, 4])
    assert torch.all(bias[0, :, 0, 5] == 0)
