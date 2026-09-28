import torch
from cifar_resnet.data import FakeCIFAR10, make_fake_cifar
from cifar_resnet.transforms import CIFARTrainTransform, random_horizontal_flip
from torch.utils.data import DataLoader


def test_fake_cifar_is_reproducible_without_network() -> None:
    first = make_fake_cifar(12, seed=11)
    second = make_fake_cifar(12, seed=11)

    x_first, y_first = first[0]
    x_second, y_second = second[0]
    assert torch.equal(x_first, x_second)
    assert y_first == y_second
    assert len(first) == 12
    assert x_first.shape == (3, 32, 32)
    assert x_first.dtype == torch.float32
    assert 0 <= float(x_first.min()) <= float(x_first.max()) <= 1


def test_fake_cifar_dataloader_batches_have_cifar_shapes() -> None:
    loader = DataLoader(make_fake_cifar(9, seed=2), batch_size=4)
    images, labels = next(iter(loader))

    assert images.shape == (4, 3, 32, 32)
    assert labels.shape == (4,)
    assert labels.dtype == torch.long


def test_transform_can_be_seeded_and_keeps_range() -> None:
    dataset = FakeCIFAR10(2, seed=3)
    image, _ = dataset[0]
    generator_a = torch.Generator().manual_seed(9)
    generator_b = torch.Generator().manual_seed(9)
    transform = CIFARTrainTransform(padding=2, crop_size=32, flip_probability=0.5)

    transformed_a = transform(image, generator=generator_a)
    transformed_b = transform(image, generator=generator_b)

    assert torch.equal(transformed_a, transformed_b)
    assert transformed_a.shape == image.shape
    assert torch.isfinite(transformed_a).all()


def test_horizontal_flip_is_not_applied_at_zero_probability() -> None:
    image = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)

    assert torch.equal(random_horizontal_flip(image, probability=0), image)
