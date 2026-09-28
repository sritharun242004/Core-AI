from __future__ import annotations
import numpy as np


def cross_entropy_numpy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy: shifts logits by max for stability."""
    z = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=-1))
    log_p_target = z[np.arange(len(labels)), labels] - logsumexp
    return float(-log_p_target.mean())


def cross_entropy_torch(logits, labels):
    import torch
    return torch.nn.functional.cross_entropy(logits, labels)
