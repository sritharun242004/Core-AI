# Solution notes — unsupervised-viz

These notes explain the reference implementation's decisions and the failure modes worth discussing in an interview. They are not a substitute for re-deriving the algorithms in the assignments.

## PCA: SVD is the stable implementation

Given a data matrix `X` with `n` rows and `d` columns, PCA first computes the feature mean

\[
\mu_j = \frac{1}{n}\sum_{i=1}^{n} X_{ij}
\]

and centers each row: `Xc = X - mu`. The implementation then computes

\[
X_c = U S V^T.
\]

The rows of `V^T` are orthonormal directions. Keeping the first `r` rows gives `Z = Xc V_r^T`, and reconstruction is `Z V_r + mu`. The variance of component `j` is `S_j^2/(n-1)`, so its explained-variance ratio is that number divided by the sum over all singular values.

It is tempting to form `Xc.T @ Xc` and diagonalize the covariance matrix. That is mathematically equivalent, but SVD avoids explicitly squaring the condition number and behaves better on ill-scaled data. The class still exposes the small API a learner needs: `fit`, `transform`, `fit_transform`, `inverse_transform`, `components_`, and `explained_variance_ratio_`.

A reconstruction test should not demand zero error after dimensionality reduction: discarded directions are supposed to be lost. The full-component reconstruction test is the useful numerical invariant; the two-component test checks that a deliberately low-rank fixture preserves nearly all its variance.

## k-means: initialization, assignment, update

For each row `x_i` and center `c_j`, the squared Euclidean distance is

\[
d_{ij}^2 = \sum_{f=1}^{d}(x_{if}-c_{jf})^2.
\]

Lloyd's loop alternates:

1. assign each row to the nearest center;
2. replace each non-empty center by the mean of its assigned rows;
3. stop when the largest center movement is at most `tol` or `max_iter` is reached.

The code uses broadcasting/einsum to form the distance matrix without a Python loop over rows. k-means++ chooses the first center randomly and samples later centers in proportion to their squared distance from the closest selected center. With `random_state=42`, the root generator creates a deterministic seed for each `n_init` run, and the lowest-inertia run wins.

### Empty clusters

An empty cluster makes the naive update `mean(X[labels == j])` invalid and yields NaN. In `KMeans`, the empty center is replaced with the row currently farthest from its assigned center. This is a small, observable policy: it gives the orphaned center a useful opportunity to claim a badly represented region on the next assignment. Duplicate rows and more requested clusters than distinct points remain finite; they are not magically evidence that the data has that many real groups.

## GMM: hard labels hide uncertainty

A Gaussian mixture models each point as coming from one of `K` Gaussian components. EM alternates:

- **E step:** compute responsibilities (posterior component probabilities) for each row;
- **M step:** update mixture weights, means, and covariance matrices using those responsibilities.

The workflow delegates that numerically delicate EM implementation to `sklearn.mixture.GaussianMixture`, but retains `predict_proba` as `soft_assignments`. `argmax` is convenient for coloring a panel; the probability vector is the better answer when a point lies between clusters.

## t-SNE and UMAP are visual maps, not truth meters

The workflow standardizes each built-in dataset once, then fits t-SNE and (if available) UMAP on those standardized rows. t-SNE preserves local neighborhood probabilities and has a perplexity hyperparameter; it does not promise that distances between far-apart islands or the axes themselves are meaningful. UMAP has a different neighbor-graph/topological objective. Neither should be judged by whether its colors make the labels look separated without checking the underlying metric and downstream task.

UMAP is imported inside `run_comparison`, not at module import time. This is intentional: the core project remains usable when the optional package is absent. The figure still reserves a fifth panel and writes a clear missing message, which is more honest than silently returning only four panels.

## The iris metric ruling

The project does not claim that unsupervised clusters equal biological species. Adjusted Rand Index (ARI) is used only as a reproducible teaching diagnostic because the built-in fixture includes reference labels:

\[
ARI = \frac{RI - E[RI]}{\max(RI)-E[RI]}.
\]

The test uses the pre-declared petal-length/width view and a fixed `random_state=42`, `n_init=10`. This is not a cherry-picked seed: it is part of the documented fixture and the multi-start setting is the algorithm's explicit robustness control. A separate smoke check shows that all-feature k-means with only one initialization varies substantially by seed, so the project refuses to elevate the handoff's all-feature `ARI >= 0.7` suggestion into a universal guarantee.

## What surprised me / common bugs

- Explained variance uses `n - 1` because it is sample covariance convention; the ratio is unchanged when all components are retained.
- `argmin` ties go to the first center. That is deterministic, but not a semantic preference.
- More clusters than distinct rows cannot create information. Empty-cluster repair prevents NaNs; it cannot invent separation.
- t-SNE's `random_state` makes a run reproducible, not universally correct.
- An absent optional dependency is a valid environment state. A labelled panel is a better teaching artifact than a hard import error.
- A colorful embedding is not a model evaluation. Always report the representation, seed, metric, and whether labels were used only for coloring.
