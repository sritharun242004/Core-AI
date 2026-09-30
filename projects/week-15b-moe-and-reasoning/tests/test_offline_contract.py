from pathlib import Path

import moe_reasoning_lab
import torch

PROJECT_ROOT = Path(__file__).parents[1]


def test_public_imports_and_source_has_no_network_or_download_path() -> None:
    assert hasattr(moe_reasoning_lab, "TinyMoE")
    assert hasattr(moe_reasoning_lab, "verify_arithmetic")
    source = "\n".join(path.read_text() for path in (PROJECT_ROOT / "src").rglob("*.py"))
    forbidden = ("requests.", "urllib.", "download=True", "http://", "https://")
    assert not any(token in source.lower() for token in forbidden)
    assert torch.cuda.is_available() is False or "cuda" not in source.lower()
