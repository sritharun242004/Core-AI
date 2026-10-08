import numpy as np
from linalg_lab.svd_compress import svd_reconstruct


def test_rank_full_reconstruction_is_exact():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 4))
    np.testing.assert_allclose(svd_reconstruct(a, k=4), a, atol=1e-10)


def test_rank_1_reconstruction_has_rank_1():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((10, 8))
    approx = svd_reconstruct(a, k=1)
    assert np.linalg.matrix_rank(approx, tol=1e-8) == 1


def test_error_decreases_monotonically_with_k():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((20, 15))
    errs = [np.linalg.norm(a - svd_reconstruct(a, k=k), "fro") for k in range(1, 15)]
    for i in range(len(errs) - 1):
        assert errs[i] >= errs[i + 1] - 1e-9


def test_k_zero_returns_zeros():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(svd_reconstruct(a, k=0), np.zeros_like(a))
