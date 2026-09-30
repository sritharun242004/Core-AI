# Applied ML capstone — 100 points

Choose a four-stage MovieLens recommender, fraud GNN+GBM comparison, ad-ranking/A-B simulator, or hierarchical forecasting study. Use a licensed/local small dataset before optional scale-up. A MovieLens experiment is not a claim to have operated YouTube-scale traffic.

| Criterion | 0–7: incomplete | 8–14: competent | 15–20: strong evidence |
|---|---|---|---|
| Production shape | Only a training notebook | Reproducible data/train/serve/eval path | Versioned contracts, idempotent data steps, failure handling, rollback and monitoring owner |
| Metrics rigor | Random leak-prone score | Appropriate holdout, ranking/forecast metrics | Temporal/entity splits, calibrated slices, confidence intervals and leakage probes |
| Systems thinking | No measured cost/latency | Local benchmark with denominators | Retrieval/ranking latency budget, memory/cost trade-offs, cold-start and capacity analysis |
| Experiments | No baseline | Strong simple baseline and one ablation | Offline plus honest randomized A/B simulation, power/guardrails and counterfactual assumptions |
| Design document | Architecture picture alone | Problem, requirements and trade-offs | Alternatives rejected with evidence, operational failure modes, data/consent risks and migration plan |

## Evidence checklist

- Candidate-generation recall before ranking quality; user/item cold-start treatment.
- No future interactions/features crossing the prediction-time boundary.
- Forecast horizons, seasonal-naive baseline and rolling-origin evaluation where relevant.
- A/B assignment at the proper randomization unit, SRM check, minimum duration and no peeking claim.
- Offline lift is not presented as measured causal business impact.

**Release gates:** no label/time leakage, no hidden denominator changes, no invented scale claims, and no personal-data exposure. Target 80/100 with every criterion at least 10 as a learning guideline.

**Demo:** follow one event through feature generation, retrieval, ranking and evaluation; show a stale-feature or unavailable-item failure path. **Stretch:** demonstrate why an apparently higher offline score loses under a predeclared operational guardrail.
