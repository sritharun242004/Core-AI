import torch
from cifar_resnet.data import make_fake_cifar
from cifar_resnet.model import resnet18
from cifar_resnet.train import evaluate, fit, train_epoch
from torch.utils.data import DataLoader


def test_train_epoch_produces_finite_loss_and_gradients() -> None:
    loader = DataLoader(make_fake_cifar(8, seed=5), batch_size=4)
    model = resnet18(num_classes=10, widths=(2, 4), blocks=(1, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    loss, accuracy = train_epoch(model, loader, optimizer, device="cpu")

    assert torch.isfinite(torch.tensor(loss))
    assert 0 <= accuracy <= 1
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_evaluate_does_not_build_a_graph() -> None:
    loader = DataLoader(make_fake_cifar(6, seed=7), batch_size=3)
    model = resnet18(num_classes=10, widths=(2, 4), blocks=(1, 1))

    loss, accuracy = evaluate(model, loader, device="cpu")

    assert torch.isfinite(torch.tensor(loss))
    assert 0 <= accuracy <= 1
    assert not torch.is_grad_enabled() or all(
        parameter.grad is None for parameter in model.parameters()
    )


def test_short_fit_smoke_is_deterministic_enough_and_returns_history() -> None:
    loader = DataLoader(make_fake_cifar(12, seed=4), batch_size=6, shuffle=False)
    model = resnet18(num_classes=10, widths=(2, 4), blocks=(1, 1))

    history = fit(model, loader, epochs=2, learning_rate=0.01, device="cpu")

    assert set(history) == {"train_loss", "train_accuracy"}
    assert len(history["train_loss"]) == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in history["train_loss"])
