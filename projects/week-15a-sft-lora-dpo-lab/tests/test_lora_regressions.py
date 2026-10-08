# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportCallIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportIndexIssue=false, reportReturnType=false, reportAssignmentType=false, reportMissingTypeArgument=false
import copy
from pathlib import Path

import pytest
import torch
from post_training_lab import (
    LoRALinear,
    TinyCausalLM,
    inject_lora,
    load_checkpoint,
    mark_only_lora_trainable,
    save_checkpoint,
    sft_loss,
)
from torch import nn


def tiny_model() -> TinyCausalLM:
    return TinyCausalLM(vocab_size=8, d_model=8, n_heads=2, max_seq_len=4)


def test_lora_inherits_base_dtype_device_and_eval_mode() -> None:
    base = nn.Linear(5, 3, dtype=torch.float64).eval()
    layer = LoRALinear(base, rank=2, alpha=3, dropout=0.5)
    assert layer.lora_A.dtype == base.weight.dtype
    assert layer.lora_B.dtype == base.weight.dtype
    assert layer.lora_A.device == base.weight.device
    assert not layer.training and not layer.dropout.training
    inputs = torch.randn(4, 5, dtype=torch.float64)
    torch.testing.assert_close(layer(inputs), base(inputs))
    with torch.no_grad():
        layer.lora_B.fill_(0.3)
    torch.testing.assert_close(
        layer(inputs), nn.functional.linear(inputs, layer.merged_weight(), base.bias)
    )


@pytest.mark.parametrize(
    "options",
    [
        {"rank": 0},
        {"rank": 1.5},
        {"dropout": -0.1},
        {"dropout": float("nan")},
        {"alpha": float("nan")},
        {"target_modules": ("missing",)},
    ],
)
def test_failed_injection_does_not_mutate_model(options: dict) -> None:
    model = tiny_model()
    before_state = copy.deepcopy(model.state_dict())
    before_parameters = list(model.parameters())
    before_flags = [parameter.requires_grad for parameter in before_parameters]
    before_modules = list(model.modules())
    with pytest.raises(ValueError):
        inject_lora(model, **options)
    assert list(model.modules()) == before_modules
    assert list(model.parameters()) == before_parameters
    assert [parameter.requires_grad for parameter in model.parameters()] == before_flags
    for name, value in model.state_dict().items():
        assert torch.equal(value, before_state[name])


def test_repeated_injection_failure_keeps_existing_adapters_trainable() -> None:
    model = inject_lora(tiny_model(), rank=2)
    before_flags = [parameter.requires_grad for parameter in model.parameters()]
    with pytest.raises(ValueError):
        inject_lora(model, rank=2)
    assert [parameter.requires_grad for parameter in model.parameters()] == before_flags


def test_injecting_additional_targets_does_not_freeze_existing_adapters() -> None:
    model = inject_lora(tiny_model(), rank=2, target_modules=("q_proj",))
    inject_lora(model, rank=2, target_modules=("v_proj",))
    assert all(
        parameter.requires_grad
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in (module.lora_A, module.lora_B)
    )


def test_trainability_uses_adapter_identity_not_parameter_name_substrings() -> None:
    model = nn.Module()
    model.not_lora_A = nn.Linear(4, 4)
    model.adapter = LoRALinear(nn.Linear(4, 4), rank=2, alpha=2)
    trainable = mark_only_lora_trainable(model)
    assert trainable == 16
    assert all(not parameter.requires_grad for parameter in model.not_lora_A.parameters())


def test_freezing_clears_stale_gradients_so_existing_optimizer_cannot_update_base() -> None:
    torch.manual_seed(20)
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    ids = torch.tensor([[1, 2, 3, 4]])
    sft_loss(model(ids[:, :-1]), ids[:, 1:]).backward()
    original_parameters = list(model.parameters())
    before = [parameter.detach().clone() for parameter in original_parameters]
    inject_lora(model, rank=2)
    assert all(parameter.grad is None for parameter in original_parameters)
    optimizer.step()
    for value, parameter in zip(before, original_parameters, strict=True):
        assert torch.equal(value, parameter)


def test_adapter_update_changes_only_adapters_and_keeps_tied_base_weights() -> None:
    torch.manual_seed(25)
    model = inject_lora(tiny_model(), rank=2)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    ids = torch.tensor([[1, 2, 3, 4]])
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        sft_loss(model(ids[:, :-1]), ids[:, 1:]).backward()
        optimizer.step()
    assert model.lm_head.weight is model.token_embedding.weight
    assert any(
        not torch.equal(value, before[name])
        for name, value in model.named_parameters()
        if "lora_" in name
    )
    assert all(
        torch.equal(value, before[name])
        for name, value in model.named_parameters()
        if "lora_" not in name
    )


def test_lora_checkpoint_restores_nonempty_optimizer_and_resumes_exactly(tmp_path: Path) -> None:
    torch.manual_seed(31)
    model = inject_lora(tiny_model(), rank=2, alpha=5)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.1)
    ids = torch.tensor([[1, 2, 3, 4]])

    def step(policy, optim):
        optim.zero_grad(set_to_none=True)
        loss = sft_loss(policy(ids[:, :-1]), ids[:, 1:])
        loss.backward()
        optim.step()
        return loss.detach()

    step(model, optimizer)
    path = tmp_path / "lora.pt"
    save_checkpoint(model, optimizer, path, step=1, description="adapter resume")
    restored = inject_lora(tiny_model(), rank=2, alpha=5)
    restored_optimizer = torch.optim.AdamW(
        [p for p in restored.parameters() if p.requires_grad], lr=0.1
    )
    metadata = load_checkpoint(restored, restored_optimizer, path)
    assert metadata == {"step": 1, "description": "adapter resume"}
    assert restored_optimizer.state_dict()["state"]
    torch.testing.assert_close(model(ids), restored(ids), rtol=0, atol=0)
    torch.testing.assert_close(
        step(model, optimizer), step(restored, restored_optimizer), rtol=0, atol=0
    )
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    assert all(
        not parameter.requires_grad
        for name, parameter in restored.named_parameters()
        if "lora_" not in name
    )


@pytest.mark.parametrize("mismatch", ["alpha", "dropout", "rank", "uninjected"])
def test_incompatible_lora_checkpoint_rejected_before_mutating_target(
    tmp_path: Path, mismatch: str
) -> None:
    source = inject_lora(tiny_model(), rank=2, alpha=5, dropout=0.2)
    source.blocks[0].attn.q_proj.lora_B.data.fill_(0.2)
    path = tmp_path / "lora.pt"
    save_checkpoint(source, path)
    target = tiny_model()
    if mismatch != "uninjected":
        inject_lora(
            target,
            rank=3 if mismatch == "rank" else 2,
            alpha=2 if mismatch == "alpha" else 5,
            dropout=0.0 if mismatch == "dropout" else 0.2,
        )
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="LoRA"):
        load_checkpoint(target, path)
    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name])


def test_checkpoint_restores_saved_trainability(tmp_path: Path) -> None:
    model = inject_lora(tiny_model(), rank=2, freeze_base=False)
    path = tmp_path / "partial-freeze.pt"
    save_checkpoint(model, path)
    restored = inject_lora(tiny_model(), rank=2)
    load_checkpoint(restored, path)
    assert [p.requires_grad for p in restored.parameters()] == [
        p.requires_grad for p in model.parameters()
    ]
