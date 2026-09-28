"""Numeric parity across NumPy / PyTorch / (where applicable) pure Python.
Review Focus #5 — the same problem must yield identical answers across impls.
Torch tests skip cleanly if torch isn't installed."""

from __future__ import annotations
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st


def test_matmul_parity_python_vs_numpy():
    from np_vs_pt.matmul import matmul_python, matmul_numpy
    a = np.random.default_rng(0).random((4, 3)).tolist()
    b = np.random.default_rng(1).random((3, 5)).tolist()
    py = matmul_python(a, b)
    numpy_ = matmul_numpy(np.array(a), np.array(b))
    np.testing.assert_allclose(np.array(py), numpy_, atol=1e-6)


def test_matmul_parity_numpy_vs_torch():
    torch = pytest.importorskip("torch")
    from np_vs_pt.matmul import matmul_numpy, matmul_torch
    a = np.random.default_rng(0).random((4, 3))
    b = np.random.default_rng(1).random((3, 5))
    numpy_ = matmul_numpy(a, b)
    torch_ = matmul_torch(torch.tensor(a), torch.tensor(b)).numpy()
    np.testing.assert_allclose(numpy_, torch_, atol=1e-6)


def test_softmax_parity_python_vs_numpy():
    from np_vs_pt.softmax import softmax_python, softmax_numpy
    x = [1.0, 2.0, 3.0, 100.0]  # 100 tests numerical stability
    py = softmax_python(x)
    npx = softmax_numpy(np.array(x))
    np.testing.assert_allclose(py, npx, atol=1e-9)
    assert abs(sum(py) - 1.0) < 1e-9


def test_softmax_parity_numpy_vs_torch():
    torch = pytest.importorskip("torch")
    from np_vs_pt.softmax import softmax_numpy, softmax_torch
    x = np.array([1.0, 2.0, 3.0, 100.0])
    npx = softmax_numpy(x)
    tx  = softmax_torch(torch.tensor(x)).numpy()
    np.testing.assert_allclose(npx, tx, atol=1e-6)


def test_cross_entropy_parity():
    torch = pytest.importorskip("torch")
    from np_vs_pt.crossentropy import cross_entropy_numpy, cross_entropy_torch
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(8, 10))
    labels = rng.integers(0, 10, size=8)
    ce_np = cross_entropy_numpy(logits, labels)
    ce_pt = cross_entropy_torch(torch.tensor(logits), torch.tensor(labels)).item()
    assert abs(ce_np - ce_pt) < 1e-5


def test_kmeans_one_step_parity():
    torch = pytest.importorskip("torch")
    from np_vs_pt.kmeans import kmeans_one_step_numpy, kmeans_one_step_torch
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 2))
    centers = rng.normal(size=(3, 2))
    new_np = kmeans_one_step_numpy(X, centers)
    new_pt = kmeans_one_step_torch(torch.tensor(X), torch.tensor(centers)).numpy()
    np.testing.assert_allclose(new_np, new_pt, atol=1e-5)


def test_topk_parity_python_vs_numpy():
    from np_vs_pt.topk import topk_python, topk_numpy
    x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    py = topk_python(x, 3)
    npx = topk_numpy(np.array(x), 3).tolist()
    assert py == npx


def test_topk_parity_numpy_vs_torch():
    torch = pytest.importorskip("torch")
    from np_vs_pt.topk import topk_numpy, topk_torch
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    npx = topk_numpy(x, 3).tolist()
    tx  = topk_torch(torch.tensor(x), 3).tolist()
    assert npx == tx


@given(n=st.integers(min_value=1, max_value=20))
@settings(max_examples=25, deadline=None)
def test_softmax_sums_to_one_across_ranks(n):
    """Property test — softmax over any shape sums to 1 within tolerance."""
    from np_vs_pt.softmax import softmax_numpy
    rng = np.random.default_rng(n * 100)
    x = rng.normal(size=n)
    y = softmax_numpy(x)
    assert abs(y.sum() - 1.0) < 1e-9
