from pathlib import Path

import torch
from mini_bpe_pretrain import (
    TinyCausalLM,
    TinyStoriesDataset,
    load_checkpoint,
    make_tinystories_corpus,
    pretrain,
    save_checkpoint,
)


def test_tiny_stories_fixture_is_in_memory_and_shifted() -> None:
    corpus = make_tinystories_corpus()
    assert len(corpus) >= 4
    dataset = TinyStoriesDataset(corpus, seq_len=12, vocab_size=48)
    inputs, targets = dataset[0]
    assert inputs.shape == targets.shape == (12,)
    assert torch.equal(inputs[1:], targets[:-1])
    assert dataset.tokenizer.vocab_size <= 48


def test_model_shapes_causality_and_position_variants() -> None:
    dataset = TinyStoriesDataset(make_tinystories_corpus(), seq_len=16, vocab_size=48)
    for position_type in ("rope", "alibi"):
        model = TinyCausalLM(
            vocab_size=dataset.tokenizer.vocab_size,
            d_model=24,
            n_heads=4,
            n_layers=1,
            max_seq_len=16,
            position_type=position_type,
        )
        x, _ = dataset[0]
        logits = model(x[None])
        changed = x.clone()
        changed[-1] = (changed[-1] + 1) % dataset.tokenizer.vocab_size
        changed_logits = model(changed[None])
        assert logits.shape == (1, 16, dataset.tokenizer.vocab_size)
        assert torch.allclose(logits[:, :-1], changed_logits[:, :-1], atol=1e-5)


def test_pretraining_reduces_fixed_batch_loss_and_is_seeded() -> None:
    dataset = TinyStoriesDataset(make_tinystories_corpus(), seq_len=16, vocab_size=48)
    torch.manual_seed(3)
    model = TinyCausalLM(
        dataset.tokenizer.vocab_size, d_model=24, n_heads=4, n_layers=1, max_seq_len=16
    )
    result = pretrain(model, dataset, steps=8, batch_size=4, learning_rate=0.02, seed=3)
    assert len(result.losses) == 8
    assert result.losses[-1] < result.losses[0]
    assert result.tokens_seen == 8 * 4 * 16


def test_checkpoint_round_trip_restores_logits_and_optimizer(tmp_path: Path) -> None:
    dataset = TinyStoriesDataset(make_tinystories_corpus(), seq_len=12, vocab_size=48)
    model = TinyCausalLM(
        dataset.tokenizer.vocab_size, d_model=16, n_heads=4, n_layers=1, max_seq_len=12
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    result = pretrain(model, dataset, steps=2, batch_size=2, optimizer=optimizer, seed=5)
    sample = dataset[0][0][None]
    expected = model(sample).detach()
    path = tmp_path / "step-2.pt"
    save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        step=result.steps,
        tokenizer=dataset.tokenizer,
        history=result.losses,
    )

    restored_model = TinyCausalLM.from_config(model.config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    metadata = load_checkpoint(path, restored_model, optimizer=restored_optimizer)
    assert metadata["step"] == 2
    assert metadata["tokenizer"]["vocab_size"] == dataset.tokenizer.vocab_size
    assert torch.allclose(expected, restored_model(sample), atol=1e-6)
