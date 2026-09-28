# Challenge — elastic-net proximal gradient

Extend the logistic objective with both L1 and L2 penalties. Implement a proximal-gradient update:

1. take a gradient step for the data loss and L2 term;
2. apply soft-thresholding to the feature weights for the L1 term;
3. never apply the threshold to the intercept;
4. plot sparsity as the L1 coefficient increases.

Compare your selected features with a coordinate-descent implementation and explain why the two optimizers can choose different but similarly predictive models.
