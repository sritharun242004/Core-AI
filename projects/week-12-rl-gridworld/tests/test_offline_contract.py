from pathlib import Path

import rl_gridworld
import torch

PROJECT_ROOT = Path(__file__).parents[1]


def test_reference_project_is_cpu_only_and_has_no_download_path() -> None:
    source = "\n".join(path.read_text() for path in (PROJECT_ROOT / "src").rglob("*.py"))
    forbidden = ("cuda", ".cuda(", "download=True", "requests.", "urllib.", "http://", "https://")
    assert not any(token in source.lower() for token in forbidden)
    assert torch.cuda.is_available() is False or "cuda" not in source.lower()


def test_public_module_imports_without_side_effects() -> None:
    assert hasattr(rl_gridworld, "Gridworld")
    assert hasattr(rl_gridworld, "train_q_learning")
    assert hasattr(rl_gridworld, "train_policy_gradient")
