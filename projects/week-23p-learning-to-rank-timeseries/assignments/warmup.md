# Warmup — 30 minutes

1. Derive RankNet gradients for tied scores and labels [1,0]. Multiply by the NDCG@2 swap delta to obtain the LambdaRank-style gradients.
2. Compute ATE, sample-variance SE and the nominal normal 95% CI for control [1,3] versus treatment [4,6]. Explain why the small sample weakens inferential claims.
3. At origin 20 with lookback 6 and horizon 3, list the prediction input and target indices. Show the latest valid training window.
4. Forecast five steps with seasonal naive from last season [4,5,6].

Deliver calculations and one finite-difference gradient check. Worked answers: `SOLUTION_NOTES.md`.
