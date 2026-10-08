import torch
from cifar_resnet.model import CIFARCNN, BasicBlock, ResNet, resnet18
from cifar_resnet.transfer import freeze_backbone, replace_classifier, unfreeze_all


def test_cifar_cnn_and_resnet_preserve_batch_and_class_shapes() -> None:
    x = torch.randn(4, 3, 32, 32)
    cnn = CIFARCNN(num_classes=10, channels=(8, 16))
    net = resnet18(num_classes=10, widths=(8, 16, 32, 64), blocks=(1, 1, 1, 1))

    assert cnn(x).shape == (4, 10)
    assert net(x).shape == (4, 10)


def test_basic_block_identity_skip_keeps_shape() -> None:
    block = BasicBlock(8, 8, stride=1)
    x = torch.randn(2, 8, 16, 16)

    y = block(x)

    assert y.shape == x.shape
    assert block.downsample is None
    assert torch.isfinite(y).all()


def test_basic_block_projection_skip_changes_resolution_and_channels() -> None:
    block = BasicBlock(8, 16, stride=2)
    y = block(torch.randn(2, 8, 16, 16))

    assert y.shape == (2, 16, 8, 8)
    assert block.downsample is not None


def test_resnet_has_a_real_skip_path_and_backpropagates() -> None:
    model = ResNet(num_classes=3, widths=(4, 8), blocks=(1, 1))
    x = torch.randn(2, 3, 32, 32)
    loss = model(x).square().mean()
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert any("downsample" in name for name, _ in model.named_parameters())


def test_resnet_zero_init_residual_starts_blocks_near_identity() -> None:
    model = resnet18(num_classes=10, widths=(4, 8, 16, 32), blocks=(1, 1, 1, 1))

    bn_weights = [block.bn2.weight for block in model.modules() if isinstance(block, BasicBlock)]

    assert bn_weights
    assert all(torch.equal(weight, torch.zeros_like(weight)) for weight in bn_weights)


def test_cnn_rejects_non_rgb_input() -> None:
    model = CIFARCNN(num_classes=10)

    try:
        model(torch.randn(2, 1, 32, 32))
    except RuntimeError as error:
        assert "channel" in str(error).lower() or "expected" in str(error).lower()
    else:
        raise AssertionError("one-channel input should not silently pass")


def test_transfer_helpers_replace_head_and_freeze_backbone() -> None:
    model = resnet18(num_classes=10, widths=(4, 8), blocks=(1, 1))

    replace_classifier(model, 4)
    freeze_backbone(model)

    assert model.fc.out_features == 4
    assert all(
        parameter.requires_grad == name.startswith("fc.")
        for name, parameter in model.named_parameters()
    )
    unfreeze_all(model)
    assert all(parameter.requires_grad for parameter in model.parameters())
