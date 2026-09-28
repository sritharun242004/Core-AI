# Compute note — Week 12

**Tier: 🟢 green (local CPU).**

The reference project is a deterministic 4×4 MDP with a 16×4 float64 Q-table
and a tiny PyTorch policy. The default checks finish in seconds on a laptop;
no accelerator, dataset, account, or external service is required. The
notebook intentionally uses only the in-memory environment.

## Reproducible reference run

```bash
cd projects/week-12-rl-gridworld
PYTHONPATH=src ../../.venv/bin/python -m pytest
../../.venv/bin/ruff check src tests
PYTHONPATH=src ../../.venv/bin/python notebooks/01-rl-gridworld.py
```

Record at least the Python/PyTorch versions, seed, episode budget, learning
rate/alpha, gamma, epsilon schedule, optimizer, and greedy evaluation episode
count. Q-learning uses NumPy's seeded generator; REINFORCE seeds PyTorch before
constructing the policy and samples only from the local environment.

## Optional extension budget

A larger map or many-seed study remains CPU-friendly, but runtime grows with
`episodes × max_steps × policy forward passes`. Keep the default 4×4 fixture as
the regression test. If changing the map, report wall coordinates, reward
scale, time limit, observation encoding, train/evaluation protocol, and all
seeds. A return on one fixed map is not a generalization benchmark.

## Resource boundary

The project does not download data, initialize an accelerator, or promise PPO
benchmark results. A cloud machine cannot make the tiny fixture more valid; use
one only for an explicitly documented larger ablation, not for the reference
checks.
