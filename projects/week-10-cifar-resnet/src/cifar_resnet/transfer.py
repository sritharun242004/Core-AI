"""Small transfer-learning helpers that keep the classifier boundary explicit."""

from __future__ import annotations

from torch import nn


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze every parameter except the common ``fc``/``classifier`` head."""

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc.") or name.startswith("classifier.")
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Make every parameter trainable again."""

    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace a model's ``fc`` or ``classifier`` while retaining its backbone."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    raise TypeError("model must expose a Linear fc or classifier head")
