# Research engineering capstone — 100 points

Choose mini-Chinchilla, mini-GRPO, matched-active-parameter MoE, or test-time-compute scaling. Start from Research α/β and Weeks 13–15b. A novel frontier result is not required; one well-controlled extension is.

| Criterion | 0–7: incomplete | 8–14: competent | 15–20: strong evidence |
|---|---|---|---|
| Correctness | No independent oracle | Core equations and shapes tested | Forward/backward, masking, state and edge-case oracles plus reference parity |
| Rigor | Cherry-picked run | Baseline, fixed holdout, logged seeds | Preregistered ablations, paired seeds/error bars, compute-matched comparisons, failed runs preserved |
| Code quality | Manual irreproducible notebook | Versioned config and one-command run | Clean environment reproduction, checkpoints/manifests, readable APIs and regression CI |
| Writeup | Unsupported score claim | Four-page structured report | Exact source attribution, reproduction/deviation table, raw artifacts and transparent limitations |
| Extension | Unmotivated extra component | One hypothesis and comparison | Discriminating controlled experiment; negative results interpreted without overclaiming |

## Evidence checklist

- Parameter/token/FLOP counting conventions and actual budget.
- Dataset version/license, splits, tokenizer and contamination limitations.
- Equations cross-checked against primary sources; DPO Stanford, GRPO DeepSeekMath then R1.
- At least three seeds where feasible, or explicit explanation of inadequate uncertainty.
- Full output artifacts rather than screenshots alone.

**Release gates:** no known objective/gradient bug, no fabricated measurements, no undisclosed data leakage, and no secrets. These gates override a high numeric self-score. A provisional target is 80/100 with no criterion below 10; it is a study rubric, not a hiring bar.

**Demo:** explain the claim in 30 seconds, show one independent oracle, one ablation and one failed/limited result. **Stretch:** preregister an experiment someone else can falsify within your budget. Do not claim that a tiny synthetic scaling fit replicates the full Chinchilla law.
