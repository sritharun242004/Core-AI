from pathlib import Path

import post_training_lab
import torch

PROJECT_ROOT = Path(__file__).parents[1]


def test_public_api_is_offline_and_tiny() -> None:
    assert hasattr(post_training_lab, "TinyCausalLM")
    assert hasattr(post_training_lab, "dpo_loss")
    source = "\n".join(path.read_text() for path in (PROJECT_ROOT / "src").rglob("*.py"))
    forbidden = ("requests.", "urllib.", "download=True", "http://", "https://")
    assert not any(token in source.lower() for token in forbidden)
    model = post_training_lab.TinyCausalLM(vocab_size=16, d_model=16, n_heads=4, max_seq_len=8)
    assert model.num_parameters() < 20_000
    assert torch.cuda.is_available() is False or "cuda" not in source.lower()
