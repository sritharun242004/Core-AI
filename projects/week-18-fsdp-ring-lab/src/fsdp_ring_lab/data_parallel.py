# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownLambdaType=false, reportCallIssue=false, reportUnnecessaryIsInstance=false, reportIndexIssue=false, reportPrivateUsage=false, reportMissingTypeArgument=false, reportUnknownParameterType=false, reportMissingImports=false, reportPossiblyUnboundVariable=false
"""Sequential CPU simulation of sample-weighted gradient reduction."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .partition import partition


@dataclass(frozen=True)
class GradientResult:
    loss: float
    gradients: tuple[Tensor, ...]


def _validate(model: nn.Module, x: Tensor, target: Tensor) -> tuple[nn.Parameter, ...]:
    if x.ndim < 2 or target.ndim < 2 or x.shape[0] == 0 or target.shape[0] != x.shape[0]:
        raise ValueError("nonempty X and target must share batch dimension")
    params = tuple(model.parameters())
    if not params or any(not p.requires_grad for p in params):
        raise ValueError("model must have trainable parameters only")
    if any(
        t.device.type != "cpu" or not t.is_floating_point() or not torch.isfinite(t).all()
        for t in (x, target, *params)
    ):
        raise ValueError("finite floating-point CPU tensors are required")
    if any(t.dtype != x.dtype for t in (target, *params)):
        raise ValueError("inputs, targets and parameters must share dtype")
    # Training batchnorm depends on batch composition; dropout changes the RNG
    # stream. Neither obeys the per-example deterministic model used here.
    for module in model.modules():
        if module.training and isinstance(
            module, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd)
        ):
            raise ValueError("training BatchNorm/Dropout is outside the equivalence contract")
    return params


def data_parallel_gradients(
    model: nn.Module, x: Tensor, target: Tensor, *, world_size: int = 2
) -> GradientResult:
    """MSE averaged across ALL target elements, not an unweighted rank mean.

    Reuses one immutable pointwise model sequentially to emulate synchronized
    replicas. Custom stateful/stochastic/batch-coupled modules are unsupported.
    Empty ranks contribute zero. Returns detached gradients, never mutates .grad.
    This simulates reduction arithmetic; it launches no worker processes.
    """
    params = _validate(model, x, target)
    specs = partition(x.shape[0], world_size)
    gradients = tuple(torch.zeros_like(p) for p in params)
    loss = 0.0
    for spec in specs:
        if spec.valid_size == 0:
            continue
        end = spec.start + spec.valid_size
        prediction = model(x[spec.start : end])
        local_target = target[spec.start : end]
        if prediction.shape != local_target.shape:
            raise ValueError("prediction and target must match exactly; broadcasting is forbidden")
        local_loss = (prediction - local_target).square().sum() / target.numel()
        if not torch.isfinite(local_loss):
            raise ValueError("MSE overflow or nonfinite model output")
        local_gradients = torch.autograd.grad(local_loss, params)
        if not all(torch.isfinite(g).all() for g in local_gradients):
            raise ValueError("nonfinite model gradient")
        gradients = tuple(
            total + local.detach() for total, local in zip(gradients, local_gradients, strict=True)
        )
        loss += local_loss.detach().item()
    return GradientResult(loss, gradients)


def full_batch_gradients(model: nn.Module, x: Tensor, target: Tensor) -> GradientResult:
    """Single full-batch mean-MSE oracle."""
    params = _validate(model, x, target)
    prediction = model(x)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must match exactly")
    loss = (prediction - target).square().mean()
    if not torch.isfinite(loss):
        raise ValueError("MSE overflow or nonfinite model output")
    gradients = torch.autograd.grad(loss, params)
    if not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError("nonfinite model gradient")
    return GradientResult(loss.detach().item(), tuple(g.detach() for g in gradients))
