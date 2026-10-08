# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportCallIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false
"""Acceptance tests for the Week 7 unsupervised-learning lab.

These tests are intentionally written before the implementation.  Run them from
this project with ``PYTHONPATH=src python -m pytest`` to see the initial RED
state.
"""

import matplotlib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score
from unsupervised_viz.kmeans import KMeans
from unsupervised_viz.pca import PCA
from unsupervised_viz.workflow import run_comparison

matplotlib.use("Agg")


def test_pca_reconstruction_and_explained_variance():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(80, 2))
    x = np.column_stack(
        [
            latent[:, 0],
            2.0 * latent[:, 1],
            latent[:, 0] - latent[:, 1],
            0.05 * rng.normal(size=80),
        ]
    )

    model = PCA(n_components=2).fit(x)
    reduced = model.transform(x)
    reconstructed = model.inverse_transform(reduced)

    assert reduced.shape == (80, 2)
    assert model.components_.shape == (2, 4)
    assert model.explained_variance_ratio_.shape == (2,)
    assert 0.0 < model.explained_variance_ratio_.sum() <= 1.0
    # The two dominant directions should preserve nearly all of this low-rank data.
    assert model.explained_variance_ratio_.sum() > 0.98
    assert np.mean((x - reconstructed) ** 2) < 0.01

    full = PCA().fit(x)
    np.testing.assert_allclose(full.inverse_transform(full.transform(x)), x, atol=1e-10)
    np.testing.assert_allclose(full.explained_variance_ratio_.sum(), 1.0, atol=1e-12)


def test_kmeans_is_reproducible_and_recovers_documented_petal_view():
    iris = load_iris()
    # The handoff's all-feature claim is not a reliable acceptance test.  This
    # deliberately documented petal-length/width view is the separable teaching fixture.
    x = iris.data[:, 2:4]
    first = KMeans(n_clusters=3, random_state=42, n_init=10).fit(x)
    second = KMeans(n_clusters=3, random_state=42, n_init=10).fit(x)

    np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_)
    np.testing.assert_array_equal(first.labels_, second.labels_)
    assert adjusted_rand_score(iris.target, first.labels_) >= 0.7
    assert np.isfinite(first.inertia_)


def test_kmeans_handles_empty_clusters_without_nan_centers():
    x = np.array([[0.0], [0.0], [10.0], [10.0]])
    model = KMeans(n_clusters=4, random_state=0, n_init=1, max_iter=20).fit(x)

    assert model.cluster_centers_.shape == (4, 1)
    assert np.isfinite(model.cluster_centers_).all()
    assert np.isfinite(model.inertia_)
    assert model.labels_.shape == (len(x),)
    np.testing.assert_array_equal(model.predict(x), model.labels_)


def test_comparison_has_four_required_methods_and_labeled_optional_umap_panel():
    result = run_comparison(
        dataset="iris",
        random_state=11,
        include_umap=True,
        make_figure=True,
    )

    assert result.dataset_name == "iris"
    assert {"pca", "kmeans", "gmm", "tsne"}.issubset(result.embeddings)
    assert set(result.embeddings["pca"].shape) == {2, 150}
    for embedding in result.embeddings.values():
        assert embedding.shape == (150, 2)
        assert np.isfinite(embedding).all()
    assert result.cluster_labels["kmeans"].shape == (150,)
    assert result.cluster_labels["gmm"].shape == (150,)
    assert len(result.figure.axes) == 5

    if result.umap_available:
        assert "umap" in result.embeddings
    else:
        assert "umap" in result.missing_methods
        assert "missing" in result.missing_methods["umap"].lower()
        assert any("missing" in axis.get_title().lower() for axis in result.figure.axes)
