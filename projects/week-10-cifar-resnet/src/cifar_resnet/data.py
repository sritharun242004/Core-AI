"""Offline fixtures and an optional real CIFAR-10 loader."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset

ImageTransform = Callable[..., Tensor]


class _CIFARDatasets(Protocol):
    def CIFAR10(  # noqa: N802
        self,
        *,
        root: Path,
        train: bool,
        download: bool,
        transform: Callable[[object], Tensor],
    ) -> Dataset[tuple[Tensor, int]]: ...


class _VisionFunctional(Protocol):
    def pil_to_tensor(self, pic: object) -> Tensor: ...


class FakeCIFAR10(Dataset[tuple[Tensor, int]]):
    """A deterministic CIFAR-shaped dataset with a learnable class signal.

    Images are generated in memory, so tests never download CIFAR-10. A subtle
    class-dependent tint and patch make the fixture useful for a short training
    smoke test without pretending to model the real data distribution.
    """

    def __init__(
        self,
        n_samples: int = 128,
        *,
        num_classes: int = 10,
        seed: int = 0,
        image_size: int = 32,
        transform: ImageTransform | None = None,
    ) -> None:
        if n_samples < 0 or num_classes <= 0 or image_size <= 0:
            raise ValueError("sample count, classes, and image size must be positive")
        self.num_classes = num_classes
        self.transform = transform
        generator = torch.Generator().manual_seed(seed)
        self.targets = torch.arange(n_samples, dtype=torch.long) % num_classes
        self.images = (
            torch.rand(
                n_samples,
                3,
                image_size,
                image_size,
                generator=generator,
                dtype=torch.float32,
            )
            * 0.15
        )
        # Encode a weak, deliberately simple signal: each class gets a channel
        # tint and a translated 4x4 patch. The background remains noisy.
        for index, target in enumerate(self.targets):
            label = int(target)
            channel = label % 3
            self.images[index, channel] += 0.55
            row = 2 + (label * 3) % max(1, image_size - 5)
            col = 2 + (label * 5) % max(1, image_size - 5)
            self.images[index, :, row : row + 4, col : col + 4] += 0.2
        self.images.clamp_(0, 1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image = self.images[index].clone()
        label = int(self.targets[index])
        if self.transform is not None:
            # A per-item seed keeps augmentation deterministic across worker
            # scheduling while retaining random-looking crops and flips.
            generator = torch.Generator().manual_seed(index)
            try:
                image = self.transform(image, generator=generator)
            except TypeError:
                image = self.transform(image)
        return image, label


def make_fake_cifar(
    n_samples: int = 128,
    *,
    num_classes: int = 10,
    seed: int = 0,
    image_size: int = 32,
    transform: ImageTransform | None = None,
) -> FakeCIFAR10:
    """Return an offline deterministic CIFAR-shaped dataset."""

    return FakeCIFAR10(
        n_samples,
        num_classes=num_classes,
        seed=seed,
        image_size=image_size,
        transform=transform,
    )


def cifar10_datasets(
    root: str | Path = "data/cifar10",
    *,
    train_transform: ImageTransform | None = None,
    download: bool = False,
) -> tuple[Dataset[tuple[Tensor, int]], Dataset[tuple[Tensor, int]]]:
    """Create torchvision CIFAR-10 datasets lazily for the optional real run.

    Importing this module and running its tests never imports torchvision or
    contacts the network. Set ``download=True`` only when explicitly opting in
    to the real dataset.
    """

    try:
        # Optional torchvision is loaded only on the explicitly requested path.
        datasets = cast(_CIFARDatasets, import_module("torchvision.datasets"))
        functional = cast(_VisionFunctional, import_module("torchvision.transforms.functional"))
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "cifar10_datasets requires optional torchvision; use make_fake_cifar for tests"
        ) from error
    root = Path(root)

    def to_tensor(image: object) -> Tensor:
        return functional.pil_to_tensor(image).float().div(255)

    real_transform = to_tensor
    if train_transform is not None:
        # torchvision's CIFAR dataset yields PIL images, while our dependency-
        # light transform deliberately works on tensors. Convert only on this
        # explicitly opted-in path; the offline fixture stays torch-only.
        def real_transform(image: object) -> Tensor:
            return train_transform(to_tensor(image))

    train = datasets.CIFAR10(root=root, train=True, download=download, transform=real_transform)
    test = datasets.CIFAR10(root=root, train=False, download=download, transform=to_tensor)
    return train, test
