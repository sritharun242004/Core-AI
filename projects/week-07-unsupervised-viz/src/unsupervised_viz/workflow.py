"""A reproducible, side-by-side comparison of unsupervised projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.datasets import load_digits, load_iris
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .kmeans import KMeans
from .pca import PCA

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass
class ComparisonOutput:
    """Outputs from :func:`run_comparison`.

    ``kmeans`` and ``gmm`` are clustering methods rather than 2-D projection
    methods, so their panels reuse the PCA coordinates and color points by the
    learned hard/soft cluster assignment.  The distinction is retained in
    ``cluster_labels`` and ``soft_assignments``.
    """

    dataset_name: str
    X: FloatArray
    y: IntArray
    embeddings: dict[str, FloatArray]
    cluster_labels: dict[str, IntArray]
    soft_assignments: FloatArray
    missing_methods: dict[str, str]
    figure: Figure | None
    umap_available: bool

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(self.embeddings)

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly view (the Matplotlib figure is omitted)."""

        return {
            "dataset_name": self.dataset_name,
            "X": self.X,
            "y": self.y,
            "embeddings": self.embeddings,
            "cluster_labels": self.cluster_labels,
            "soft_assignments": self.soft_assignments,
            "missing_methods": self.missing_methods,
            "umap_available": self.umap_available,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow notebook code to use ``result["embeddings"]`` as well as attributes."""

        return self.as_dict()[key]


def run_comparison(
    dataset: str = "digits",
    *,
    random_state: int = 42,
    n_clusters: int | None = None,
    max_samples: int | None = None,
    include_umap: bool = True,
    make_figure: bool = True,
    tsne_perplexity: float = 30.0,
) -> ComparisonOutput:
    """Run PCA, k-means, GMM, t-SNE, and optional UMAP on one dataset.

    The built-in ``digits`` and ``iris`` datasets require no network access.
    ``max_samples`` is useful for a quick notebook run; sampling is fixed by
    ``random_state``.  UMAP is deliberately imported only when requested, so
    the project remains installable without the optional ``umap-learn`` extra.
    """

    dataset_name, x, y = _load_dataset(dataset)
    rng = np.random.default_rng(random_state)
    if max_samples is not None:
        if not isinstance(max_samples, int) or max_samples < 2:
            raise ValueError("max_samples must be an integer >= 2")
        if max_samples < len(x):
            indices = np.sort(rng.choice(len(x), size=max_samples, replace=False))
            x, y = x[indices], y[indices]

    if n_clusters is None:
        n_clusters = int(np.unique(y).size)
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")

    # Scaling is fitted on this one local dataset, then shared by the two
    # clustering models.  This keeps iris' centimeter units from dominating.
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2).fit(x_scaled)
    pca_embedding = pca.transform(x_scaled)

    kmeans_model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    ).fit(x_scaled)
    gmm_model = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        n_init=5,
        random_state=random_state,
    ).fit(x_scaled)
    gmm_labels = gmm_model.predict(x_scaled).astype(int)

    safe_perplexity = min(float(tsne_perplexity), max(1.0, (len(x) - 1) / 3.0))
    tsne_embedding = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=750,
        random_state=random_state,
    ).fit_transform(x_scaled)

    embeddings: dict[str, FloatArray] = {
        "pca": pca_embedding,
        # Clustering does not itself produce a 2-D coordinate system.  Reusing
        # PCA makes the cluster boundary comparison honest and interpretable.
        "kmeans": pca_embedding,
        "gmm": pca_embedding,
        "tsne": tsne_embedding,
    }
    cluster_labels: dict[str, IntArray] = {
        "kmeans": kmeans_model.labels_.astype(int),
        "gmm": gmm_labels,
    }
    missing_methods: dict[str, str] = {}
    umap_available = False

    if include_umap:
        try:
            import umap  # type: ignore[import-not-found]

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, len(x) - 1),
                min_dist=0.1,
                metric="euclidean",
                random_state=random_state,
            )
            embeddings["umap"] = reducer.fit_transform(x_scaled)
            umap_available = True
        except Exception as exc:  # optional dependency may be absent or broken
            missing_methods["umap"] = (
                "UMAP missing/unavailable; install the optional 'umap' extra "
                f"to enable it ({type(exc).__name__}: {exc})"
            )

    figure = plot_comparison(
        ComparisonOutput(
            dataset_name=dataset_name,
            X=x,
            y=y.astype(int),
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            soft_assignments=gmm_model.predict_proba(x_scaled),
            missing_methods=missing_methods,
            figure=None,
            umap_available=umap_available,
        ),
        make_figure=make_figure,
    )
    output = ComparisonOutput(
        dataset_name=dataset_name,
        X=x,
        y=y.astype(int),
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        soft_assignments=gmm_model.predict_proba(x_scaled),
        missing_methods=missing_methods,
        figure=figure,
        umap_available=umap_available,
    )
    return output


def plot_comparison(result: ComparisonOutput, *, make_figure: bool = True) -> Figure | None:
    """Plot one panel per required method and a labeled UMAP-missing panel."""

    if not make_figure:
        return None

    panel_order = ["pca", "kmeans", "gmm", "tsne"]
    if "umap" in result.embeddings or "umap" in result.missing_methods:
        panel_order.append("umap")
    figure, axes = plt.subplots(
        1,
        len(panel_order),
        figsize=(4.3 * len(panel_order), 4.0),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes[0]
    for axis, method in zip(flat_axes, panel_order, strict=True):
        if method in result.missing_methods:
            axis.set_title("UMAP (missing)")
            axis.text(
                0.5,
                0.5,
                result.missing_methods[method],
                ha="center",
                va="center",
                wrap=True,
                fontsize=9,
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            continue

        colors = result.y
        title = method.upper()
        if method in result.cluster_labels:
            colors = result.cluster_labels[method]
            title += " • cluster IDs"
        elif method == "pca" or method == "tsne":
            title += " • labels"
        axis.scatter(
            result.embeddings[method][:, 0],
            result.embeddings[method][:, 1],
            c=colors,
            cmap="tab10",
            s=18,
            alpha=0.78,
            linewidths=0,
        )
        axis.set_title(title)
        axis.set_xlabel("component 1")
        axis.set_ylabel("component 2")
        axis.grid(alpha=0.18)
    figure.suptitle(f"Unsupervised comparison · {result.dataset_name}")
    return figure


def _load_dataset(dataset: str) -> tuple[str, FloatArray, IntArray]:
    normalized = dataset.lower().strip()
    if normalized == "iris":
        bundle = load_iris()
    elif normalized == "digits":
        bundle = load_digits()
    else:
        raise ValueError("dataset must be one of: 'iris', 'digits'")
    return (
        normalized,
        np.asarray(bundle.data, dtype=float),
        np.asarray(bundle.target, dtype=int),
    )


# A short alias reads well in exploratory notebooks.
compare = run_comparison
