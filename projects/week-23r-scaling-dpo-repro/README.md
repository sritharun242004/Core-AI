# Track R beta — Scaling and DPO reproduction

Three actual small causal fixed-context MLP language models train on disjoint synthetic context IDs and report held-out loss, parameter count and processed input tokens. A separate known-law fixture tests log-linear fitting. A tiny one-token completion policy demonstrates Stanford DPO against an immutable reference. All runs are offline.

```bash
cd projects/week-23r-scaling-dpo-repro
uv run --extra dev pytest
uv run python notebooks/01-scaling-and-preferences.py
```

APIs: `run_three_sizes`, `fit_power_law`, `paired_seed_interval`, `dpo_loss`, `train_preference`. The fitter assumes fixed irreducible loss E in L(N)=E+A*N^-alpha. It permits negative fitted slopes instead of fabricating improvements. It does not identify a joint parameter/data law from three points.

## Honest mini-Chinchilla extension

Hoffmann et al. 2022 is DeepMind research. Read the Epoch AI replication caveat. The shipped smoke is **not** a Transformer/TinyStories reproduction and its exponent is not a Chinchilla estimate. For the optional real experiment, use the Week 14 decoder with widths yielding three logged model sizes, a licensed/versioned local TinyStories corpus, a fixed tokenizer and held-out stories. Sweep data budgets at each size, multiple seeds and matched compute where required. Record N, D, measured wall time, optimizer schedule, held-out loss and uncertainty. Three sizes alone cannot identify all coefficients of E+A/N^alpha+B/D^beta; fix assumptions or collect more independent budgets.

DPO is Rafailov et al. 2023, Stanford. The one-token policy exercises the exact logistic objective but lacks language, human feedback and broad safety evaluation. Week 15a supplies completion masks and causal sequence scoring for a real decoder. Never present the toy as an aligned LLM.
