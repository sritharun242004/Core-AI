# Solutions and research notes

With fixed E, log(L-E)=log A-alpha log N. Build a design matrix [1,-log N] and solve least squares. All excess losses must be positive and at least three distinct sizes are required by this exercise. A perfect fit to a generated law proves algebra, not empirical scaling. Report the sensitivity of alpha to E and the residuals; extrapolation beyond observed sizes is not warranted.

The three model runs use disjoint context IDs and identical data budget across widths. Larger widths can generalize worse on this small task. Equal tokens are not equal FLOPs; the original Chinchilla question concerns compute-optimal allocation of parameters and training tokens. Do not force monotonic curves or claim independent data/model exponents from a single one-dimensional sweep.

DPO loss is -log sigmoid(beta*((log pi_chosen-log pi_rejected)-(log ref_chosen-log ref_rejected))). At a zero margin the loss is log 2. Its chosen-score derivative is negative; rejected derivative positive. Reference scores are detached and the copied reference has gradients disabled. All four score tensors must be finite: an infinite chosen score must not masquerade as zero loss. Fixture helpers seed only the CPU default generator inside an RNG-restoring context, preserving accelerator streams. The completion here is one token, so sum/mean coincide. For sequences, use summed completion log-probabilities and correct causal masks.

Bootstrap paired seed differences, not unrelated run rows. Two seeds technically allow an interval but not a strong claim. Preregister seeds, metrics, selection policy and compute ceiling. The four-page report should contain claim, method, reproduction table, ablation, limits and one justified extension.
