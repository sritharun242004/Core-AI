from pathlib import Path

import torch
from post_training_lab import (
    TinyCausalLM,
    load_checkpoint,
    make_sft_examples,
    save_checkpoint,
    train_sft,
)


def test_sft_training_is_deterministic_and_reduces_loss(tmp_path: Path) -> None:
    torch.manual_seed(10)
    model = TinyCausalLM(vocab_size=16, d_model=16, n_heads=4, max_seq_len=8)
    history = train_sft(model, make_sft_examples(), epochs=5, learning_rate=0.05)
    assert len(history["loss"]) == 5
    assert history["loss"][-1] < history["loss"][0]
    checkpoint = tmp_path / "toy.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    save_checkpoint(model, optimizer, checkpoint, step=5, metrics={"loss": history["loss"][-1]})
    restored = TinyCausalLM(vocab_size=16, d_model=16, n_heads=4, max_seq_len=8)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.05)
    metadata = load_checkpoint(restored, restored_optimizer, checkpoint)
    assert metadata["step"] == 5
    for left, right in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(left, right)


def test_checkpoint_parent_directory_is_created(tmp_path: Path) -> None:
    model = TinyCausalLM(vocab_size=8, d_model=8, n_heads=2, max_seq_len=4)
    target = tmp_path / "nested" / "checkpoint.pt"
    save_checkpoint(model, path=target)
    assert target.exists()
