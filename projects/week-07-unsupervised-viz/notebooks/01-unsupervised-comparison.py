# %% [markdown]
# # Unsupervised learning: one dataset, five views
#
# This notebook is percent-format so it can run in Jupyter or be converted with
# jupytext. The required panels are NumPy PCA, NumPy k-means, sklearn GMM, and
# sklearn t-SNE. UMAP is optional and appears as a labelled missing panel when
# `umap-learn` is not installed.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score

from unsupervised_viz import KMeans, PCA, run_comparison

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. NumPy PCA
#
# The implementation centers the matrix and uses the right singular vectors.
# The iris view below uses standardization only for a fair feature-scale
# comparison; PCA itself does not depend on sklearn.

# %%
iris = load_iris()
X = iris.data.astype(float)
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

pca = PCA(n_components=2).fit(X_scaled)
Z = pca.transform(X_scaled)
reconstructed = pca.inverse_transform(Z)

print("components:", pca.components_.shape)
print("explained variance ratio:", pca.explained_variance_ratio_)
print("cumulative variance:", pca.explained_variance_ratio_.sum())
print("reconstruction MSE:", np.mean((X_scaled - reconstructed) ** 2))

# %% [markdown]
# ## 2. NumPy k-means and the documented iris view
#
# Petal length and width provide the separable teaching view used by the
# acceptance test. ARI uses the fixture's labels only as an offline diagnostic;
# k-means itself never sees them.

# %%
X_petal = iris.data[:, 2:4]
clusters = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_petal)
ari = adjusted_rand_score(iris.target, clusters.labels_)
print("cluster centers:\n", clusters.cluster_centers_)
print("inertia:", clusters.inertia_)
print("petal-view ARI:", ari)

# %% [markdown]
# ## 3. The comparative workflow

# %%
iris_result = run_comparison("iris", random_state=42, include_umap=True)
iris_result.figure.savefig(OUTPUT / "iris-comparison.png", dpi=160)
print("iris methods:", iris_result.methods)
print("optional methods missing:", iris_result.missing_methods)

# %% [markdown]
# GMM's `soft_assignments` retains uncertainty that hard colors hide.

# %%
responsibilities = iris_result.soft_assignments
margin = np.sort(responsibilities, axis=1)[:, -1] - np.sort(responsibilities, axis=1)[:, -2]
uncertain_rows = np.argsort(margin)[:5]
print("most ambiguous rows:")
for row in uncertain_rows:
    print(row, responsibilities[row], "margin=", margin[row])

# %% [markdown]
# ## 4. Digits at a notebook-friendly sample size
#
# t-SNE is a visualization, not a classifier. The sample limit makes iteration
# quick; remove it when you want the full built-in fixture.

# %%
digits_result = run_comparison(
    "digits",
    random_state=42,
    max_samples=900,
    include_umap=True,
)
digits_result.figure.savefig(OUTPUT / "digits-comparison-900.png", dpi=160)
print("digits rows:", len(digits_result.X))
print("digits methods:", digits_result.methods)
print("optional methods missing:", digits_result.missing_methods)

# %% [markdown]
# ## 5. Close figures in batch runs

# %%
plt.close("all")
