import torch
from post_training_lab import (
    LoRALinear,
    TinyCausalLM,
    count_parameters,
    dequantize_int8,
    inject_lora,
    mark_only_lora_trainable,
    qlora_recipe,
    quantize_int8,
)
from torch import nn


def test_lora_replaces_targets_freezes_base_and_starts_as_identity() -> None:
    torch.manual_seed(4)
    model = TinyCausalLM(vocab_size=16, d_model=16, n_heads=4, max_seq_len=8)
    before = model(torch.tensor([[1, 2, 3, 4]])).detach()
    total = count_parameters(model)
    inject_lora(model, rank=2, alpha=4, target_modules=("q_proj", "v_proj"))
    trainable = mark_only_lora_trainable(model)
    after = model(torch.tensor([[1, 2, 3, 4]])).detach()
    assert isinstance(model.blocks[0].attn.q_proj, LoRALinear)
    assert trainable < total
    assert trainable == count_parameters(model, trainable_only=True)
    assert torch.allclose(before, after)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    )


def test_int8_round_trip_and_recipe_are_explicit() -> None:
    weight = torch.tensor([[-1.0, 0.0, 0.5], [0.25, 0.75, -0.3]])
    packed = quantize_int8(weight)
    restored = dequantize_int8(packed)
    assert packed.values.dtype == torch.int8
    assert restored.shape == weight.shape
    assert torch.max(torch.abs(restored - weight)) < 0.02
    recipe = qlora_recipe()
    assert recipe["base_weights"] == "int8 teaching representation"
    assert "LoRA" in recipe["trainable"]


def test_lora_linear_has_expected_low_rank_parameter_count() -> None:
    layer = LoRALinear(nn.Linear(6, 4), rank=2, alpha=2)
    assert (
        sum(parameter.numel() for parameter in layer.parameters() if parameter.requires_grad) == 20
    )
