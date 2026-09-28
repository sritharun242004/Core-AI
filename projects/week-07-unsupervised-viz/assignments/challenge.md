# Challenge — stress the unsupervised story (3+ hours)

Choose one of the following research-engineering challenges. Start with three failing tests and keep the data split/metric decisions in writing.

## A. Initialization and stability study

Add a `stability_report` helper that runs k-means across 20 declared seeds and returns inertia, iteration count, and ARI **only for the documented iris petal view**. Plot the distribution. Add a test that the report is deterministic and a test that an empty-cluster fixture never emits NaN. Explain why choosing the best inertia run is not the same as proving that the clusters are real.

## B. Streaming-scale PCA

Implement a chunked PCA experiment for a synthetic matrix that does not fit comfortably in a single temporary copy. Compare the covariance/SVD approach with a randomized or incremental method, report reconstruction error, and document the numerical trade-off. Do not call a library result “from scratch”; identify which steps you delegated.

## C. Cluster validation without labels

Implement silhouette score for a small dataset using only NumPy, then compare it with ARI on iris. Include a deliberately overlapping synthetic fixture where silhouette and ARI disagree. The write-up must say which metric is available in a real unlabeled deployment and why the labeled iris score is only a teaching diagnostic.

**Staff-level review questions:**

- Which conclusions survive a different seed, distance metric, or feature scaling?
- How would you detect a representation that makes one large cluster and many tiny artifacts?
- Why is a t-SNE picture not a substitute for a downstream retrieval or classification metric?
- What would you log so a future reader can reproduce the exact panel and optional dependency state?
