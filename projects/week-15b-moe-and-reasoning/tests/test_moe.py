# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportCallIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportIndexIssue=false, reportReturnType=false, reportAssignmentType=false
import pytest
import torch
from moe_reasoning_lab.moe import TinyMoE, TinyMoEClassifier
from moe_reasoning_lab.training import make_toy_supervised_data, train_supervised_moe


def test_router_top_k_weights_sum_to_one_and_indices_are_unique() -> None:
    torch.manual_seed(0)
    model = TinyMoE(d_model=8, hidden_dim=16, num_experts=4, top_k=2)
    x = torch.randn(3, 5, 8)
    routing = model.route(x)
    assert routing.topk_indices.shape == (15, 2)
    assert routing.topk_weights.shape == (15, 2)
    assert torch.allclose(routing.topk_weights.sum(-1), torch.ones(15))
    assert torch.all(routing.topk_indices[:, 0] != routing.topk_indices[:, 1])
    assert torch.all((routing.topk_indices >= 0) & (routing.topk_indices < 4))


def test_expert_output_and_moe_output_shapes() -> None:
    torch.manual_seed(1)
    model = TinyMoE(d_model=8, hidden_dim=12, num_experts=3, top_k=2)
    x = torch.randn(2, 7, 8)
    output, auxiliary = model(x, return_aux=True)
    assert output.shape == x.shape
    assert auxiliary.ndim == 0
    assert all(expert(torch.randn(4, 8)).shape == (4, 8) for expert in model.experts)


def test_capacity_drops_are_explicit_and_load_balance_is_finite() -> None:
    torch.manual_seed(2)
    model = TinyMoE(
        d_model=8,
        hidden_dim=12,
        num_experts=4,
        top_k=2,
        capacity_factor=0.25,
    )
    routing = model.route(torch.randn(5, 6, 8))
    assert routing.capacity == 4  # ceil(30 tokens * 2 slots * 0.25 / 4)
    assert routing.dispatch_mask.shape == (30, 2)
    assert routing.dispatch_mask.sum().item() < 60
    assert routing.expert_loads.sum() == routing.dispatch_mask.sum()
    assert torch.all(routing.expert_loads <= routing.capacity)
    assert torch.isfinite(routing.aux_loss)
    assert torch.isfinite(model(torch.randn(5, 6, 8))).all()


def test_toy_supervised_moe_reduces_loss() -> None:
    torch.manual_seed(3)
    features, labels = make_toy_supervised_data(n_samples=32, seed=3)
    model = TinyMoEClassifier(input_dim=2, d_model=16, hidden_dim=24, num_experts=3)
    result = train_supervised_moe(model, features, labels, steps=45, learning_rate=0.03)
    assert result.losses[-1] < result.losses[0]
    assert result.losses[-1] < 0.7 * result.losses[0]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_balance_loss_uses_pre_capacity_load_not_capped_counts() -> None:
    model = TinyMoE(d_model=2, num_experts=2, top_k=1, capacity_factor=0.1)
    with torch.no_grad():
        model.router.gate.weight.copy_(torch.tensor([[8.0, 0.0], [-8.0, 0.0]]))
    routing = model.route(torch.ones(10, 2))
    assert routing.expert_loads.tolist() == [1.0, 0.0]
    assert routing.selected_loads.tolist() == [10.0, 0.0]
    assert float(routing.aux_loss.detach()) > 1.9  # collapse remains visible after drops
    routing.aux_loss.backward()
    assert model.router.gate.weight.grad is not None
    assert torch.isfinite(model.router.gate.weight.grad).all()


def test_dispatch_matches_manual_weighted_expert_sum_without_drops() -> None:
    torch.manual_seed(8)
    model = TinyMoE(d_model=4, num_experts=3, top_k=2, capacity_factor=3.0)
    x = torch.randn(2, 3, 4)
    output = model(x)
    route = model.last_routing
    flat = x.reshape(-1, 4)
    expected = torch.zeros_like(flat)
    for token in range(6):
        for slot in range(2):
            expert = model.experts[int(route.topk_indices[token, slot])]
            expected[token] += route.topk_weights[token, slot] * expert(flat[token])
    assert route.dispatch_mask.all()
    assert torch.allclose(output.reshape(-1, 4), expected, atol=1e-6)


def test_fully_dropped_token_has_zero_expert_branch_output() -> None:
    model = TinyMoE(d_model=2, num_experts=2, top_k=1, capacity_factor=0.1)
    with torch.no_grad():
        model.router.gate.weight.copy_(torch.tensor([[8.0, 0.0], [-8.0, 0.0]]))
    output = model(torch.ones(1, 10, 2))
    assert torch.equal(output[0, 1:], torch.zeros(9, 2))


@pytest.mark.parametrize("top_k", [0, 5])
def test_router_rejects_invalid_top_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        TinyMoE(num_experts=4, top_k=top_k)


def test_router_rejects_empty_input_and_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="capacity_factor"):
        TinyMoE(capacity_factor=0)
    with pytest.raises(ValueError, match="token"):
        TinyMoE(d_model=2).route(torch.empty(0, 2))
