"""Week 9 mini-torch: NumPy tensors, reverse-mode autodiff, and an MLP."""

from .data import load_mnist_shaped, make_mnist_shaped, synthetic_mnist
from .engine import Tensor
from .losses import cross_entropy, mse_loss
from .nn import MLP, SGD, Linear, accuracy, fit, train_mlp

__all__ = [
    "MLP",
    "SGD",
    "Linear",
    "Tensor",
    "accuracy",
    "cross_entropy",
    "fit",
    "load_mnist_shaped",
    "make_mnist_shaped",
    "mse_loss",
    "synthetic_mnist",
    "train_mlp",
]
