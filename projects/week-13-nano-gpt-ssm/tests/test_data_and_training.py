import math

import pytest
import torch
from nano_gpt_ssm import (
    CharVocab,
    NanoGPT,
    TinySSMLanguageModel,
    language_model_loss,
    make_tiny_dataset,
    training_step,
)


def test_fixture_is_deterministic_with_shifted_targets() -> None:
    first = make_tiny_dataset(seq_len=12)
    second = make_tiny_dataset(seq_len=12)
    x, y = first[0]
    assert first.vocab.itos == tuple(sorted(set(first.text)))
    assert first.vocab.itos == second.vocab.itos
    assert torch.equal(x, second[0][0])
    assert x.dtype == y.dtype == torch.long
    assert x.shape == y.shape == (12,)
    assert torch.equal(x[1:], y[:-1])
    assert first.vocab.decode(x.tolist()) == first.text[:12]
    assert first.vocab.decode(y.tolist()) == first.text[1:13]


def test_vocabulary_rejects_unknown_characters() -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        CharVocab("abc").encode("z")


def test_transformer_output_shape_and_tied_embedding() -> None:
    model = NanoGPT(13, d_model=16, n_heads=4, n_layers=2, max_seq_len=12)
    assert model(torch.randint(13, (3, 12))).shape == (3, 12, 13)
    assert model.lm_head.weight is model.token_embedding.weight
    assert len(model.blocks) == 2


def test_transformer_is_causal_across_multiple_blocks() -> None:
    torch.manual_seed(11)
    model = NanoGPT(9, d_model=12, n_heads=3, n_layers=2, max_seq_len=10).eval()
    ids = torch.randint(9, (2, 10))
    changed = ids.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 9
    assert torch.allclose(model(ids)[:, :6], model(changed)[:, :6], atol=1e-6)


def test_uniform_logits_have_log_vocabulary_cross_entropy() -> None:
    logits = torch.zeros(2, 3, 5)
    targets = torch.zeros(2, 3, dtype=torch.long)
    losses = language_model_loss(logits, targets, reduction="none")
    assert losses.shape == (2, 3)
    assert torch.allclose(losses, torch.full((2, 3), math.log(5)))
    assert torch.allclose(losses.sum(), language_model_loss(logits, targets, reduction="sum"))
    assert torch.allclose(losses.mean(), language_model_loss(logits, targets))


@pytest.mark.parametrize("architecture", ["transformer", "ssm"])
def test_gradient_and_short_fixed_batch_loss_reduction(architecture: str) -> None:
    torch.manual_seed(12)
    dataset = make_tiny_dataset(seq_len=12)
    inputs = torch.stack([dataset[index][0] for index in range(4)])
    targets = torch.stack([dataset[index][1] for index in range(4)])
    if architecture == "transformer":
        model = NanoGPT(dataset.vocab.size, d_model=16, n_heads=2, n_layers=1)
    else:
        model = TinySSMLanguageModel(dataset.vocab.size, embedding_size=12, state_size=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
    initial = float(language_model_loss(model(inputs), targets).detach())
    training_step(model, inputs, targets, optimizer)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(torch.count_nonzero(p.grad) for p in model.parameters())
    for _ in range(39):
        training_step(model, inputs, targets, optimizer)
    model.eval()
    final = float(language_model_loss(model(inputs), targets).detach())
    assert final < initial * 0.5, (initial, final)


def test_generation_is_greedy_and_restores_training_mode() -> None:
    torch.manual_seed(13)
    model = NanoGPT(5, d_model=8, n_heads=2, n_layers=1, max_seq_len=4)
    prompt = torch.tensor([[1, 2, 3]])
    first = model.generate(prompt, max_new_tokens=5)
    second = model.generate(prompt, max_new_tokens=5)
    assert model.training
    assert first.shape == (1, 8)
    assert torch.equal(first, second)
    assert torch.equal(first[:, :3], prompt)


def test_transformer_rejects_excess_context() -> None:
    model = NanoGPT(5, d_model=8, n_heads=2, max_seq_len=4)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.ones(1, 5, dtype=torch.long))


@pytest.mark.parametrize("architecture", ["transformer", "ssm"])
def test_models_reject_empty_ids(architecture: str) -> None:
    model = NanoGPT(5) if architecture == "transformer" else TinySSMLanguageModel(5)
    with pytest.raises(ValueError, match="time"):
        model(torch.empty(2, 0, dtype=torch.long))
