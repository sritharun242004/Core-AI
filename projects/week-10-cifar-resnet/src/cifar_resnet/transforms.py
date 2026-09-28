"""Small tensor-only CIFAR augmentations; torchvision is not required for tests."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def random_horizontal_flip(
    image: Tensor,
    probability: float = 0.5,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Flip a CHW image using a supplied generator for reproducible tests."""

    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if torch.rand((), generator=generator).item() < probability:
        return image.flip(-1)
    return image


def random_crop(
    image: Tensor,
    size: int = 32,
    padding: int = 4,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Pad a CHW image then select a random square crop."""

    if image.ndim != 3:
        raise ValueError("image must have shape (channels, height, width)")
    if size <= 0 or padding < 0:
        raise ValueError("size must be positive and padding cannot be negative")
    padded = functional.pad(image, (padding, padding, padding, padding))
    max_offset_h = padded.shape[-2] - size
    max_offset_w = padded.shape[-1] - size
    if max_offset_h < 0 or max_offset_w < 0:
        raise ValueError("crop size cannot exceed padded image dimensions")
    top = int(torch.randint(max_offset_h + 1, (), generator=generator).item())
    left = int(torch.randint(max_offset_w + 1, (), generator=generator).item())
    return padded[:, top : top + size, left : left + size]


class CIFARTrainTransform:
    """Random crop + flip transform that accepts a deterministic generator."""

    def __init__(
        self,
        *,
        padding: int = 4,
        crop_size: int = 32,
        flip_probability: float = 0.5,
    ) -> None:
        self.padding = padding
        self.crop_size = crop_size
        self.flip_probability = flip_probability

    def __call__(
        self,
        image: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        cropped = random_crop(
            image,
            size=self.crop_size,
            padding=self.padding,
            generator=generator,
        )
        return random_horizontal_flip(
            cropped,
            probability=self.flip_probability,
            generator=generator,
        )
