import pytest
import torch
from transformer_repro import TinyTransformer, attention, causal_batch, train_fixture


def test_attention_matches_uniform_average_and_causal_prefix():
    q = torch.zeros(1, 3, 4, dtype=torch.float64)
    v = torch.tensor([[[1.0], [3.0], [8.0]]], dtype=torch.float64)
    actual, weights = attention(q, q, v)
    torch.testing.assert_close(actual[..., 0], torch.tensor([[1, 2, 4]], dtype=torch.float64))
    assert torch.equal(weights.triu(1), torch.zeros_like(weights))
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 3, dtype=torch.float64))


def test_attention_gradient_check():
    torch.manual_seed(2)
    tensors = tuple(torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True) for _ in range(3))
    assert torch.autograd.gradcheck(lambda q, k, v: attention(q, k, v)[0], tensors)


def test_future_tokens_cannot_change_prefix_logits():
    torch.manual_seed(3)
    model = TinyTransformer(vocab_size=8, width=16, heads=2)
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[1, 2, 7, 6]])
    torch.testing.assert_close(model(first)[:, :2], model(second)[:, :2])


def test_batch_is_exactly_shifted_and_has_no_random_future_label():
    inputs, targets = causal_batch(torch.tensor([0, 1, 2, 3, 4, 5]), context=3)
    torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])
    assert inputs.shape == (3, 3)
    assert targets[-1, -1] == 5


@pytest.mark.parametrize("width,heads", [(7, 2), (0, 1), (8, 0)])
def test_bad_head_shapes_raise(width, heads):
    with pytest.raises(ValueError):
        TinyTransformer(width=width, heads=heads)


def test_seeded_training_decreases_fixture_loss():
    first = train_fixture(seed=4, steps=30)
    second = train_fixture(seed=4, steps=30)
    assert first == second
    assert first[-1] < first[0] * 0.5


def test_half_precision_attention_accumulates_without_overflow():
    q = torch.full((1, 2, 64), 40.0, dtype=torch.float16, requires_grad=True)
    k = q.detach().clone().requires_grad_(True)
    v = torch.ones(1, 2, 3, dtype=torch.float16, requires_grad=True)
    out, _ = attention(q, k, v)
    torch.testing.assert_close(out, torch.ones_like(out))
    out.sum().backward()
    assert all(torch.isfinite(value.grad).all() for value in (q, k, v))


def test_training_does_not_reseed_accelerator_generators(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    monkeypatch.setattr(torch.mps, "manual_seed", lambda seed: calls.append(seed))
    before = torch.random.get_rng_state().clone()
    train_fixture(steps=1)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not calls


def test_checkpoint_roundtrip_is_exact(tmp_path):
    model = TinyTransformer()
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    clone = TinyTransformer()
    clone.load_state_dict(torch.load(path, weights_only=True))
    x = torch.tensor([[0, 1, 2]])
    torch.testing.assert_close(model(x), clone(x))
