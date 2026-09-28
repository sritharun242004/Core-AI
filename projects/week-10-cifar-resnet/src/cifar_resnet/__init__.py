"""Week 10: CNNs, residual networks, augmentation, and transfer learning."""

from .data import FakeCIFAR10, cifar10_datasets, make_fake_cifar
from .model import CIFARCNN, BasicBlock, ResNet, count_parameters, resnet18
from .train import choose_device, evaluate, fit, seed_everything, train_epoch
from .transfer import freeze_backbone, replace_classifier, unfreeze_all
from .transforms import CIFARTrainTransform, random_crop, random_horizontal_flip

__all__ = [
    "CIFARCNN",
    "BasicBlock",
    "CIFARTrainTransform",
    "FakeCIFAR10",
    "ResNet",
    "choose_device",
    "cifar10_datasets",
    "count_parameters",
    "evaluate",
    "fit",
    "freeze_backbone",
    "make_fake_cifar",
    "random_crop",
    "random_horizontal_flip",
    "replace_classifier",
    "resnet18",
    "seed_everything",
    "train_epoch",
    "unfreeze_all",
]
