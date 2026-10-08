import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from linalg_lab.vectors import cosine_similarity, dot, project


def test_dot_of_orthogonal_is_zero():
    assert dot(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_dot_of_parallel_equals_product_of_norms():
    a = np.array([3.0, 4.0])
    assert dot(a, a) == pytest.approx(25.0)


def test_cosine_of_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_of_opposite_vectors_is_negative_one():
    v = np.array([1.0, 2.0])
    assert cosine_similarity(v, -v) == pytest.approx(-1.0)


def test_cosine_of_zero_vector_is_zero_by_convention():
    z = np.array([0.0, 0.0])
    assert cosine_similarity(z, np.array([1.0, 0.0])) == 0.0


def test_project_v_onto_x_axis_zeros_y():
    v = np.array([3.0, 4.0])
    x = np.array([1.0, 0.0])
    np.testing.assert_allclose(project(v, x), np.array([3.0, 0.0]))


@given(
    st.integers(min_value=1, max_value=100).flatmap(
        lambda n: st.tuples(
            st.lists(st.floats(-10, 10, allow_nan=False), min_size=n, max_size=n),
            st.lists(st.floats(-10, 10, allow_nan=False), min_size=n, max_size=n),
        )
    )
)
def test_dot_is_commutative(vs: tuple[list[float], list[float]]) -> None:
    a, b = np.array(vs[0]), np.array(vs[1])
    assert dot(a, b) == pytest.approx(dot(b, a), abs=1e-9)
