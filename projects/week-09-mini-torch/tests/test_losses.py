from __future__ import annotations

import numpy as np
import pytest


def test_cross_entropy_reductions_have_expected_relationship() -> None:
    from mini_torch import Tensor, cross_entropy

    logits = Tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]], requires_grad=True)
    targets = np.array([0, 2])
    none = cross_entropy(logits, targets, reduction="none")
    total = cross_entropy(logits, targets, reduction="sum")
    mean = cross_entropy(logits, targets, reduction="mean")

    assert none.shape == (2,)
    assert total.item() == pytest.approx(float(none.data.sum()))
    assert mean.item() == pytest.approx(float(none.data.mean()))


def test_cross_entropy_gradient_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    from mini_torch import Tensor, cross_entropy

    values = np.array([[1.5, -0.2, 0.4], [-0.3, 0.7, 1.1]], dtype=np.float64)
    labels = np.array([2, 1])
    logits = Tensor(values, requires_grad=True)
    ours = cross_entropy(logits, labels)
    ours.backward()

    theirs_logits = torch.tensor(values, requires_grad=True)
    theirs = torch.nn.functional.cross_entropy(
        theirs_logits, torch.tensor(labels, dtype=torch.long), reduction="mean"
    )
    theirs.backward()

    assert ours.item() == pytest.approx(theirs.item(), rel=1e-9)
    np.testing.assert_allclose(logits.grad, theirs_logits.grad.numpy(), rtol=1e-8, atol=1e-9)


def test_loss_rejects_unknown_reduction() -> None:
    from mini_torch import Tensor, cross_entropy

    with pytest.raises(ValueError, match="reduction"):
        cross_entropy(Tensor([[1.0, 2.0]]), [1], reduction="median")
