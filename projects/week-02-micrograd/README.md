# micrograd — Week 2 reference project

A ~200-LOC scalar autodiff engine (Karpathy-style) plus a tiny MLP that overfits XOR.

## Prereqs

- Python 3.13, uv installed (see repo root).
- `torch` is optional (used only for one parity test).

## Run

```bash
cd projects/week-02-micrograd
uv sync --extra dev
uv run pytest -v
```

## What's inside

- `src/micrograd/engine.py` — `Value` scalar with `+`, `-`, `*`, `/`, `**`, `relu`, `exp`, `log`, `backward()`.
- `src/micrograd/nn.py` — `Neuron`, `Layer`, `MLP` — a functional NN layer, no framework.
- `notebooks/01-autodiff-tour.py` — a live walkthrough: forward, backward, GD on XOR.
- `assignments/{warmup,build,challenge}.md` — three sizes of exercise.

## Why re-implement autograd?

Because when PyTorch surprises you, you need a mental model of what `.backward()` is doing. This project fits in your head.
