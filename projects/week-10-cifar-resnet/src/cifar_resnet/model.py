"""Small CIFAR-sized CNN and ResNet models implemented with PyTorch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class ConvBNReLU(nn.Sequential):
    """A readable convolutional block used by the baseline CNN."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CIFARCNN(nn.Module):
    """A compact convolutional classifier for 32x32 RGB images.

    Adaptive average pooling means the classifier stays shape-safe for nearby
    image sizes while the default stem is tuned for CIFAR's three channels.
    """

    def __init__(
        self,
        num_classes: int = 10,
        channels: Sequence[int] = (32, 64, 128),
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must contain at least one width")
        layers: list[nn.Module] = []
        previous = in_channels
        for index, width in enumerate(channels):
            layers.append(ConvBNReLU(previous, width, stride=1 if index == 0 else 2))
            previous = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(previous, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(torch.flatten(x, start_dim=1))


class BasicBlock(nn.Module):
    """The two 3x3 convolutions and identity/projection skip from ResNet."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        *,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        if zero_init_residual:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet(nn.Module):
    """CIFAR-friendly ResNet made from :class:`BasicBlock` stages.

    Unlike ImageNet ResNet-18, the stem uses a 3x3 stride-1 convolution and no
    max-pool, preserving useful detail in 32x32 images.
    """

    def __init__(
        self,
        num_classes: int = 10,
        *,
        in_channels: int = 3,
        widths: Sequence[int] = (64, 128, 256, 512),
        blocks: Sequence[int] = (2, 2, 2, 2),
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        if len(widths) != len(blocks) or not widths:
            raise ValueError("widths and blocks must be non-empty and have equal length")
        if any(width <= 0 for width in widths) or any(count <= 0 for count in blocks):
            raise ValueError("widths and blocks must contain positive integers")

        self.inplanes = widths[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        for index, (width, count) in enumerate(zip(widths, blocks, strict=True)):
            stages.append(
                self._make_layer(
                    width,
                    count,
                    stride=1 if index == 0 else 2,
                    zero_init_residual=zero_init_residual,
                )
            )
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[-1] * BasicBlock.expansion, num_classes)

        self._initialize_weights()
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.zeros_(module.bn2.weight)

    def _make_layer(
        self,
        width: int,
        count: int,
        *,
        stride: int,
        zero_init_residual: bool,
    ) -> nn.Sequential:
        layers = [
            BasicBlock(
                self.inplanes,
                width,
                stride,
                zero_init_residual=zero_init_residual,
            )
        ]
        self.inplanes = width * BasicBlock.expansion
        layers.extend(
            BasicBlock(
                self.inplanes,
                width,
                zero_init_residual=zero_init_residual,
            )
            for _ in range(1, count)
        )
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, start_dim=1))


def resnet18(
    num_classes: int = 10,
    *,
    in_channels: int = 3,
    widths: Sequence[int] = (64, 128, 256, 512),
    blocks: Sequence[int] = (2, 2, 2, 2),
) -> ResNet:
    """Build a CIFAR ResNet-18, with widths overridable for fast experiments."""

    return ResNet(
        num_classes=num_classes,
        in_channels=in_channels,
        widths=widths,
        blocks=blocks,
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters, useful when comparing the baseline and residual net."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
