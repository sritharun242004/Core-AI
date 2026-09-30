"""Finite, dependency-free preference objectives for the toy policy."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


def sft_loss(logits: Tensor, labels: Tensor, *, ignore_index: int = -100) -> Tensor:
    """Token cross-entropy for supervised fine-tuning."""

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be [batch, time, vocab] and labels [batch, time]")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid = flat_labels != ignore_index
    # Select before softmax: even nonfinite logits at ignored positions must
    # not poison the loss or backward pass. An empty sum retains autograd.
    if not valid.any():
        return flat_logits[valid].sum()
    return functional.cross_entropy(flat_logits[valid], flat_labels[valid])


def sequence_logprob(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor | None = None,
    *,
    normalize: bool = False,
) -> Tensor:
    """Score already aligned targets; no causal shift is performed here.

    The default sums token log-probabilities. ``normalize=True`` returns their
    mean over selected targets (not padded length), as used by IPO/ORPO/SimPO.
    ``-100`` is always ignored, including when a binary mask is supplied.
    Empty sums are zero; an empty mean raises rather than scoring no response.
    """

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits must be [batch, time, vocab] and labels [batch, time]")
    valid = labels != -100
    if mask is not None:
        if mask.shape != labels.shape:
            raise ValueError("mask must have the same shape as labels")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must be binary")
        mask = mask.to(device=labels.device, dtype=torch.bool)
        valid = valid & mask
    lengths = valid.sum(dim=-1)
    if normalize and (lengths == 0).any():
        raise ValueError("normalization requires at least one selected token per sequence")
    safe_labels = labels.masked_fill(~valid, 0)
    safe_logits = logits.masked_fill(~valid.unsqueeze(-1), 0.0)
    token_logps = functional.log_softmax(safe_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    sums = token_logps.masked_fill(~valid, 0.0).sum(dim=-1)
    return sums / lengths if normalize else sums


def _as_vector(value: Tensor | float) -> Tensor:
    return value if isinstance(value, Tensor) else torch.tensor(float(value))


def _check_shapes(*values: Tensor) -> None:
    if not values[0].numel() or any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("log-probabilities must have equal, non-empty shapes")


def _reference_margin(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor | None,
    reference_rejected_logps: Tensor | None,
) -> Tensor:
    _check_shapes(policy_chosen_logps, policy_rejected_logps)
    margin = policy_chosen_logps - policy_rejected_logps
    if (reference_chosen_logps is None) != (reference_rejected_logps is None):
        raise ValueError("provide both reference chosen and rejected log-probabilities")
    if reference_chosen_logps is not None and reference_rejected_logps is not None:
        _check_shapes(policy_chosen_logps, reference_chosen_logps, reference_rejected_logps)
        margin = margin - (reference_chosen_logps.detach() - reference_rejected_logps.detach())
    return margin


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor | None = None,
    reference_rejected_logps: Tensor | None = None,
    *,
    beta: float = 0.1,
) -> Tensor:
    """Stanford Rafailov et al.'s DPO logistic preference loss.

    Inputs are summed completion log-probabilities. Reference tensors are
    detached; omitting both reference scores explicitly selects a reference-free
    variant, not standard DPO. No tokenizer or trainer is hidden here.
    """

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    margin = _reference_margin(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    return -functional.logsigmoid(beta * margin).mean()


def dpo_batch_loss(
    policy: nn.Module,
    reference: nn.Module,
    chosen_ids: Tensor,
    rejected_ids: Tensor,
    *,
    beta: float = 0.1,
    chosen_mask: Tensor | None = None,
    rejected_mask: Tensor | None = None,
) -> Tensor:
    """Compute summed causal DPO scores for full prompt+completion ids.

    Masks mark target tokens in the original ids and shift along with labels.
    Without masks every token after the first is scored; prepend a prompt/BOS
    to score the first response token. Right padding is supported by masking
    pad targets, not by an attention mask. The reference runs in eval/no-grad
    mode temporarily; its original module modes are restored even on failure.
    """

    for ids, mask in ((chosen_ids, chosen_mask), (rejected_ids, rejected_mask)):
        if ids.ndim != 2 or ids.shape[1] < 2 or ids.shape[0] == 0:
            raise ValueError("each batch must contain sequences with at least two tokens")
        if mask is not None and mask.shape != ids.shape:
            raise ValueError("mask must have the same shape as full token ids")
    if chosen_ids.shape[0] != rejected_ids.shape[0]:
        raise ValueError("chosen and rejected batch shapes must have equal batch size")
    reference_parameters = {id(parameter) for parameter in reference.parameters()}
    if any(id(parameter) in reference_parameters and parameter.requires_grad
           for parameter in policy.parameters()):
        raise ValueError("policy and reference must not share trainable parameters")

    def score(model: nn.Module, ids: Tensor, mask: Tensor | None) -> Tensor:
        target_mask = mask[:, 1:] if mask is not None else None
        if target_mask is not None and not target_mask.to(dtype=torch.bool).any(dim=-1).all():
            raise ValueError("each sequence must have at least one selected target token")
        return sequence_logprob(model(ids)[:, :-1], ids[:, 1:], target_mask)

    modes = [(module, module.training) for module in reference.modules()]
    try:
        reference.eval()
        with torch.no_grad():
            reference_chosen = score(reference, chosen_ids, chosen_mask)
            reference_rejected = score(reference, rejected_ids, rejected_mask)
    finally:
        for module, training in modes:
            module.training = training
    policy_chosen = score(policy, chosen_ids, chosen_mask)
    policy_rejected = score(policy, rejected_ids, rejected_mask)
    return dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=beta)


def kto_loss(
    policy_logps: Tensor,
    reference_logps: Tensor,
    desirable: Tensor,
    *,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> Tensor:
    """KTO's bounded logistic utility with an explicitly toy KL estimator.

    Use summed completion log-probabilities and binary desirability labels.
    The baseline is max(0, mean(policy - reference)), detached. Reusing the
    labeled batch is an approximation, NOT the paper's mismatched-pair KL
    estimator. Reference scores receive no gradients.
    """

    _check_shapes(policy_logps, reference_logps, desirable)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if any(not math.isfinite(weight) or weight < 0
           for weight in (desirable_weight, undesirable_weight)):
        raise ValueError("KTO weights must be finite and non-negative")
    if not torch.all((desirable == 0) | (desirable == 1)):
        raise ValueError("desirable labels must be binary")
    desirable = desirable.to(device=policy_logps.device, dtype=torch.bool)
    delta = policy_logps - reference_logps.detach()
    baseline = delta.detach().mean().clamp_min(0.0)
    # sigmoid(-x) = 1 - sigmoid(x), without subtractive cancellation.
    good = torch.sigmoid(beta * (baseline - delta))
    bad = torch.sigmoid(beta * (delta - baseline))
    return torch.where(
        desirable, desirable_weight * good, undesirable_weight * bad
    ).mean()


def ipo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor | None = None,
    reference_rejected_logps: Tensor | None = None,
    *,
    beta: float = 0.1,
) -> Tensor:
    """IPO's squared target margin on mean completion log-probabilities.

    Call ``sequence_logprob(..., normalize=True)`` for policy and reference.
    Reference scores are detached; omitting them is a reference-free variant.
    """

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    margin = _reference_margin(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    return (margin - 1.0 / (2.0 * beta)).square().mean()


def orpo_loss(
    chosen_logps: Tensor,
    rejected_logps: Tensor,
    sft_loss: Tensor | float = 0.0,
    *,
    beta: float = 0.1,
) -> Tensor:
    """ORPO on mean completion log-probabilities, plus an explicit SFT anchor.

    For log p, log odds is log p - log(1 - exp(log p)), not log p itself.
    Scores must be floating-point and non-positive; exact p=1 is clamped only
    at the numerical boundary. The default zero anchor computes ONLY the
    preference term; pass chosen SFT loss for ORPO.
    ``beta`` weights that term (called lambda in the paper).
    """

    _check_shapes(chosen_logps, rejected_logps)
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    if not chosen_logps.is_floating_point() or not rejected_logps.is_floating_point():
        raise ValueError("ORPO log-probabilities must be floating-point")
    for scores in (chosen_logps, rejected_logps):
        if torch.isnan(scores).any() or (scores > 0).any():
            raise ValueError("ORPO log-probabilities must be non-positive")
    # expm1 avoids cancellation when log p is close to zero. A saturated
    # p=1 log-probability is nudged inside the domain to avoid log(0).
    upper = -torch.finfo(chosen_logps.dtype).eps
    chosen_scores = chosen_logps.clamp_max(upper)
    rejected_scores = rejected_logps.clamp_max(upper)
    chosen_odds = chosen_scores - torch.log(-torch.expm1(chosen_scores))
    rejected_odds = rejected_scores - torch.log(-torch.expm1(rejected_scores))
    log_odds_ratio = chosen_odds - rejected_odds
    anchor = _as_vector(sft_loss).to(log_odds_ratio)
    return anchor.mean() + beta * (-functional.logsigmoid(log_odds_ratio)).mean()


def simpo_loss(
    chosen_logps: Tensor,
    rejected_logps: Tensor,
    *,
    beta: float = 2.0,
    gamma: float = 0.5,
    chosen_lengths: Tensor | None = None,
    rejected_lengths: Tensor | None = None,
) -> Tensor:
    """SimPO: -log sigmoid(beta * (mean_chosen - mean_rejected) - gamma).

    Inputs must already be token means, or supply both selected-token lengths
    to normalize sums. ``gamma`` is a reward-space margin, NOT gamma/beta.
    """

    _check_shapes(chosen_logps, rejected_logps)
    if not math.isfinite(beta) or beta <= 0 or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("beta must be finite and positive; gamma finite and non-negative")
    if (chosen_lengths is None) != (rejected_lengths is None):
        raise ValueError("provide both chosen and rejected lengths")
    if chosen_lengths is not None and rejected_lengths is not None:
        _check_shapes(chosen_logps, chosen_lengths, rejected_lengths)
        if any(not torch.isfinite(lengths).all() or (lengths <= 0).any()
               for lengths in (chosen_lengths, rejected_lengths)):
            raise ValueError("lengths must be finite and positive")
        chosen_lengths = chosen_lengths.to(
            device=chosen_logps.device, dtype=chosen_logps.dtype
        )
        rejected_lengths = rejected_lengths.to(
            device=rejected_logps.device, dtype=rejected_logps.dtype
        )
        chosen_logps = chosen_logps / chosen_lengths
        rejected_logps = rejected_logps / rejected_lengths
    return -functional.logsigmoid(beta * (chosen_logps - rejected_logps) - gamma).mean()
