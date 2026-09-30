# Build — exact arithmetic across artificial ranks (2–3 hr)

Implement the public API from tests before reading its implementation.

1. Split seven examples across three ranks, including valid counts [3,3,1]. Show a
   deliberately incorrect equal-rank-mean gradient, then fix it with n_r/N weighting.
   Compare every parameter gradient and one SGD update to a full-batch oracle.
2. Implement the online maximum/denominator/numerator recurrence for attention. Keep
   queries local and visit K/V owners in ring order. Use global causal positions.
3. Test W=1, W=3, W>N, nondivisible N, one-token input, future-only first blocks,
   an all-padded example, and extreme finite logits. Disallow NaNs and score overflow.
4. Compare outputs **and Q/K/V gradients with a nonuniform upstream tensor** against
   an independent dense `torch.softmax` oracle. A `.sum()` loss alone can hide bugs.
   Run gradcheck on a tiny float64 input. Explain dtype-dependent tolerance choices.
5. Execute `notebooks/01-fsdp-ring.py` offline and produce an analytic wire ledger.

**Acceptance:** float64 output absolute error <=1e-12 and gradient absolute error
<=2e-12 on the supplied tiny fixtures (with documented relative tolerance); finite
large-logit outputs; exact zero padded output/gradient; invalid contracts rejected.
No speedup, reduced-backward-memory, or real-network claim is accepted from this test.

**Submit:** tests, code, notebook output, and a one-page account of one failure you
caught. `SOLUTION_NOTES.md` sections 3–5 contain reference derivations.
