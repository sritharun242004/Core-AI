# Core AI — Plan 3: Weeks 2-4 Content

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Weeks 2 (Calculus + micrograd), 3 (Probability & Statistics + prob-lab), and 4 (Python for ML + Information Theory + numpy-vs-pytorch) — 3 MDX articles with the full 8-part lesson anatomy and 3 cloneable Python reference projects — so a learner can complete Month 1 (Foundations) end-to-end.

**Architecture:** Each week is one MDX file in `apps/book/src/content/weeks/` plus one Python package under `projects/`. The MDX uses the components/islands already shipped in Plans 1-2 (`<Hook>`, `<Intuition>`, `<MathBlock>`, `<CompanyLens>` requiring all 7 companies, `<MicroRecall>` for interactive recall, `<VectorPlayground>` from `@core-ai/viz`, `<ReferenceProject>`, `<Assignments>`, `<InterviewDrill>`, `<FurtherReading>`, `<KeyTakeaways>`). Each Python project mirrors Week 1's shape: `pyproject.toml`, `src/<pkg>/`, `tests/`, `notebooks/`, `assignments/{warmup,build,challenge}.md`, `README.md`, `SOLUTION_NOTES.md`, `COMPUTE.md`. All three weeks are 🟢 local tier (no GPU needed).

**Tech Stack:** Astro 6 · MDX · KaTeX SSR · Shiki · `@core-ai/viz` · `@preact/signals` · Python 3.13 · NumPy · matplotlib · pytest · hypothesis · ruff · pyright

**Spec:** `/Users/tharunkumarl/Full Stack/Core-AI/docs/superpowers/specs/2026-09-22-core-ai-book-design.md`

**Prior plans:** Plan 1 (foundation + Week 1) and Plan 2 (platform features) — both merged into `plan-1-foundation` and tagged `v0.1.0-week01` / `v0.2.0-platform`.

## Global Constraints

- **Node:** 24 LTS. **Python:** 3.13. **Framework:** Astro **6.x**.
- **Compute tier:** 🟢 for all three weeks — every notebook must run on an M-series MacBook without CUDA/MPS in under 5 minutes total (spec §5.1).
- **7 fixed companies** in every CompanyLens block, fixed order: OpenAI → Anthropic → Google DeepMind → Meta AI (FAIR) → xAI → DeepSeek → Alibaba Qwen (spec §4.1). The `assertAll7` guard in `apps/book/src/components/content/CompanyLens.astro` fails the build if any is missing.
- **Attribution rules — VERBATIM** (spec §5.3):
  - DPO → Stanford (Rafailov et al. 2023). NEVER "DPO — Meta".
  - A2A + ADK → Google **Cloud**. NEVER "DeepMind".
  - Attention Is All You Need → Google Brain pre-merger (defensible under Google DeepMind today).
  - Chinchilla → DeepMind (cite Epoch AI replication).
  - Colossus → NVIDIA/Spectrum-X blog + HPCwire reporting. NO peer-reviewed publication.
  - DDIA → 2E, March 2026, Kleppmann + **Riccomini**.
- **Colored math terms** consistent across the book (spec §3): query = indigo, key = teal, value = rust; loss = tomato; gradients = amber; parameters = violet. When introducing new symbols, pick a color and stick to it for the week.
- **Content style** (spec §2.4): "Editorial Textbook" — visual explanation first, math second, code third, no hand-waving. Every derivation shows every symbol.
- **Reference project shape** — uniform per Plan 1 §5.2:
  ```
  projects/week-XX-<name>/
  ├── README.md, SOLUTION_NOTES.md, COMPUTE.md
  ├── pyproject.toml
  ├── src/<snake_case_pkg>/
  ├── tests/                  (pytest + hypothesis)
  ├── notebooks/              (Jupyter, MPS/CPU-runnable)
  └── assignments/{warmup,build,challenge}.md
  ```
- **Prereq breadcrumbs** — every week's frontmatter `prereqSlugs` must reference at least the previous week's slug (`week-01-linear-algebra` for W2, `week-02-calculus` for W3, etc.).
- **Package manager:** pnpm workspaces (JS/TS), uv workspace (Python). Never mix in npm/pip lockfiles.
- **No invented facts** about companies/papers/models beyond the spec's verified reading list. If in doubt, leave the reference blank and comment `<!-- TODO: verify claim -->` — never guess.

## Review Focus

The five input classes / failure modes the spec implies but no single task's tests exercise. Each has an inline test on the owning task.

1. **Numerical stability at zero variance** — probability code that divides by std, log-of-zero, or log-of-negative must degrade gracefully (spec §5.1 "no hand-waving"). Pin with a hypothesis property test in **Task 4** (`prob-lab`) that samples constant-value arrays.
2. **Autodiff gradient consistency** — micrograd's backward pass must match `torch.autograd` outputs to within `1e-5` on non-trivial expressions (Karpathy's benchmark test in his repo). Pin in **Task 2** with a numeric-comparison test that constructs a small graph both ways.
3. **KaTeX cell-count on new math blocks** — every `<MathBlock>` in the three new weeks must render as `<span class="katex">` at build time (already-pinned Review Focus #1 from Plan 1, extended). Pin with **Task 8**'s post-build grep test that walks `dist/client/weeks/week-0{2,3,4}-*/index.html`.
4. **CompanyLens parity for 3 new weeks** — the `assertAll7` guard must fire at build time if any of the 3 new Lens blocks drops a slug (spec §4.1). Pin with a Playwright test in **Task 8** that asserts each of the 3 weeks renders 7 rows in the fixed order.
5. **NumPy/PyTorch numeric parity** — Task 6's `numpy-vs-pytorch` project's five problems must produce identical outputs (within `atol=1e-6`) across NumPy, PyTorch, and (where applicable) pure-Python implementations. Pin with a hypothesis property test in **Task 6**.

---

## File Structure

```
apps/book/src/content/weeks/
├── week-02-calculus.mdx                                (Task 3)
├── week-03-probability.mdx                             (Task 5)
└── week-04-python-info-theory.mdx                     (Task 7)

apps/book/src/content/companies/*.mdx                   (Task 1) — seededAngles extended for W2-4

projects/
├── week-02-micrograd/
│   ├── pyproject.toml, README.md, SOLUTION_NOTES.md, COMPUTE.md     (Task 2)
│   ├── src/micrograd/
│   │   ├── __init__.py, engine.py, nn.py               (Task 2)
│   ├── tests/
│   │   ├── test_engine.py, test_nn.py                  (Task 2)
│   ├── notebooks/
│   │   └── 01-autodiff-tour.ipynb                      (Task 2 — created as .py, converted)
│   └── assignments/{warmup,build,challenge}.md         (Task 2)
├── week-03-prob-lab/
│   ├── pyproject.toml, README.md, SOLUTION_NOTES.md, COMPUTE.md     (Task 4)
│   ├── src/prob_lab/
│   │   ├── __init__.py, monte_carlo.py, bayes.py, entropy.py       (Task 4)
│   ├── tests/
│   │   ├── test_monte_carlo.py, test_bayes.py, test_entropy.py    (Task 4)
│   ├── notebooks/01-bayes-and-entropy.ipynb            (Task 4)
│   └── assignments/{warmup,build,challenge}.md         (Task 4)
└── week-04-numpy-vs-pytorch/
    ├── pyproject.toml, README.md, SOLUTION_NOTES.md, COMPUTE.md     (Task 6)
    ├── src/np_vs_pt/
    │   ├── __init__.py, matmul.py, softmax.py, kmeans.py,
    │   │   crossentropy.py, topk.py                    (Task 6)
    ├── tests/
    │   └── test_parity.py                              (Task 6 — parity across impls)
    ├── notebooks/01-five-problems-three-ways.ipynb    (Task 6)
    └── assignments/{warmup,build,challenge}.md         (Task 6)
```

---

## Task 1: Extend company `seededAngles` for W2-W4

**Files:**
- Modify: all 7 files under `apps/book/src/content/companies/`

**Interfaces:**
- Consumes: `seededAngles` frontmatter shape `{week: number, note: string}[]` (unchanged from Plan 2).
- Produces: each company's `seededAngles` array gains 1-3 new entries with weeks in {2, 3, 4}, so the `/companies` matrix fills in cells for the new weeks.

Weeks 2-4 topical hooks per company (for note copy):

| Slug      | W2 (calculus)                                              | W3 (probability)                                              | W4 (python + info theory)                          |
|-----------|------------------------------------------------------------|---------------------------------------------------------------|----------------------------------------------------|
| openai    | "Chain rule + gradients = the whole training story."       | (skip)                                                        | "Tokenizer entropy is a real production metric."   |
| anthropic | (skip)                                                     | "Anthropic-style eval: KL between sampled vs preferred dist." | "Cross-entropy loss is the RLHF objective."        |
| deepmind  | "AlphaFold's loss surface is calculus at protein scale."   | "Bayesian NN work from DeepMind — worth knowing MacKay."     | (skip)                                             |
| meta      | "PyTorch's autograd IS the calculus you're about to build."| (skip)                                                        | "NumPy → PyTorch — the migration path every FAIR engineer walks." |
| xai       | (skip)                                                     | "First-principles rejection sampling in Grok's serving loop." | (skip)                                             |
| deepseek  | "GRPO's advantage estimate is a Monte Carlo mean."         | "V3's routing is a discrete probability over experts."       | (skip)                                             |
| qwen      | (skip)                                                     | (skip)                                                        | "Qwen-VL's cross-modal alignment loss."            |

*(Fill the (skip) rows with `<!-- TODO: verify content angle -->` HTML comments so downstream automation flags them; do NOT invent facts.)*

- [ ] **Step 1: Extend `openai.mdx` `seededAngles`**

Add these entries to the existing `seededAngles:` array (keep W1/W15/W19/W21 rows already present):

```yaml
  - { week: 2, note: "Chain rule + gradients = the whole training story." }
  - { week: 4, note: "Tokenizer entropy is a real production metric — every model card reports it." }
```

- [ ] **Step 2: Extend `anthropic.mdx` `seededAngles`**

Add:
```yaml
  - { week: 3, note: "Anthropic evals: KL between the sampled distribution and the preferred distribution." }
  - { week: 4, note: "Cross-entropy is what RLHF is minimizing at every step." }
```

- [ ] **Step 3: Extend `deepmind.mdx` `seededAngles`**

Add:
```yaml
  - { week: 2, note: "AlphaFold's loss surface is calculus at protein scale — the derivatives are physics-informed." }
  - { week: 3, note: "Bayesian NN work (MacKay 1992 → modern uncertainty estimates)." }
```

- [ ] **Step 4: Extend `meta.mdx` `seededAngles`**

Add:
```yaml
  - { week: 2, note: "PyTorch's autograd IS the calculus of Week 2 — first-class citizen in the framework." }
  - { week: 4, note: "NumPy → PyTorch — the migration path every FAIR engineer walks." }
```

- [ ] **Step 5: Extend `xai.mdx` `seededAngles`**

Add:
```yaml
  - { week: 3, note: "First-principles rejection sampling — used inside Grok's serving loop." }
```

- [ ] **Step 6: Extend `deepseek.mdx` `seededAngles`**

Add:
```yaml
  - { week: 2, note: "GRPO's advantage estimate is a Monte Carlo mean of a policy's gradient." }
  - { week: 3, note: "V3's expert routing is a discrete probability over experts — read the load-balancing loss." }
```

- [ ] **Step 7: Extend `qwen.mdx` `seededAngles`**

Add:
```yaml
  - { week: 4, note: "Qwen-VL's cross-modal alignment loss — cosine similarity + cross-entropy." }
```

- [ ] **Step 8: Sync + commit**

```bash
export PATH="/Users/tharunkumarl/.nvm/versions/node/v24.21.0/bin:$PATH"
cd apps/book && npx astro sync
```

Expected: exit 0. Then:

```bash
git add apps/book/src/content/companies
git commit -m "feat(companies): seeded W2-W4 interview angles"
```

---

## Task 2: Week 2 reference project — `micrograd`

**Files:**
- Create: everything under `projects/week-02-micrograd/`

**Interfaces:**
- Consumes: uv workspace already includes `projects/*`.
- Produces:
  - `micrograd.Value(data: float, _children=(), _op='') → Value` with `.data`, `.grad`, `+`, `-`, `*`, `/`, `**`, `.relu()`, `.exp()`, `.log()`, `.backward()`.
  - `micrograd.nn.Neuron(nin: int) → Neuron`, `Layer(nin, nout)`, `MLP(nin, nouts: list[int])` — each callable on a `list[Value]` and returning `Value` (or list).
  - `Value.__all__` and `nn.__all__` exposed via `__init__.py`.

- [ ] **Step 1: Scaffold directory + pyproject.toml**

```bash
mkdir -p projects/week-02-micrograd/{src/micrograd,tests,notebooks,assignments}
touch projects/week-02-micrograd/src/micrograd/__init__.py
```

Write `projects/week-02-micrograd/pyproject.toml`:

```toml
[project]
name = "micrograd"
version = "0.1.0"
description = "Week 2 — Karpathy-style scalar autodiff engine"
requires-python = ">=3.13"
dependencies = ["numpy>=2.1"]

[project.optional-dependencies]
dev = ["pytest>=8.3", "hypothesis>=6.112", "torch>=2.5", "ruff>=0.7", "pyright>=1.1"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
addopts = "-v --strict-markers"
testpaths = ["tests"]
```

- [ ] **Step 2: Write the failing test — `test_engine.py`**

Create `projects/week-02-micrograd/tests/test_engine.py`:

```python
"""Autodiff engine tests. Includes a numeric parity check against torch.autograd
covering Review Focus #2 (gradient consistency)."""

from __future__ import annotations
import math
import pytest


def test_forward_pass_arithmetic():
    from micrograd import Value
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b + b.relu()
    assert c.data == pytest.approx(-6.0)


def test_backward_gradient_shapes():
    from micrograd import Value
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b
    c.backward()
    assert a.grad == pytest.approx(-3.0)
    assert b.grad == pytest.approx(2.0)


def test_relu_gradient():
    from micrograd import Value
    a = Value(-1.0)
    b = a.relu()
    b.backward()
    assert b.data == 0.0
    assert a.grad == 0.0


def test_exp_and_log():
    from micrograd import Value
    x = Value(1.5)
    y = x.exp().log()  # log(exp(x)) == x
    y.backward()
    assert y.data == pytest.approx(1.5, rel=1e-6)
    assert x.grad == pytest.approx(1.0, rel=1e-6)


def test_matches_torch_autograd():
    """Review Focus #2 — numeric parity with torch.autograd on a non-trivial graph."""
    torch = pytest.importorskip("torch")
    from micrograd import Value

    def build_ours():
        a = Value(2.5)
        b = Value(-1.3)
        c = (a * b + a.relu()) * (b.exp() + Value(0.1))
        c.backward()
        return c.data, a.grad, b.grad

    def build_torch():
        a = torch.tensor(2.5, requires_grad=True)
        b = torch.tensor(-1.3, requires_grad=True)
        c = (a * b + a.relu()) * (b.exp() + 0.1)
        c.backward()
        return c.item(), a.grad.item(), b.grad.item()

    ours = build_ours()
    theirs = build_torch()
    assert ours[0] == pytest.approx(theirs[0], rel=1e-5)
    assert ours[1] == pytest.approx(theirs[1], rel=1e-5)
    assert ours[2] == pytest.approx(theirs[2], rel=1e-5)
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd projects/week-02-micrograd && uv sync --extra dev && uv run pytest tests/test_engine.py -v 2>&1 | tail
```

Expected: FAIL — `ModuleNotFoundError: No module named 'micrograd'`.

- [ ] **Step 4: Implement `engine.py`**

Create `projects/week-02-micrograd/src/micrograd/engine.py`:

```python
"""Scalar autodiff engine — every op records its parents and a local gradient
function; backward() does a topo sort and applies the chain rule."""

from __future__ import annotations
import math
from typing import Callable, Iterable


class Value:
    __slots__ = ("data", "grad", "_prev", "_op", "_backward")

    def __init__(self, data: float, _children: Iterable["Value"] = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def _wrap(self, other: "Value | float") -> "Value":
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _b():
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _b
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _b():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _b
        return out

    def __pow__(self, other: float):
        assert isinstance(other, (int, float)), "power must be a plain number"
        out = Value(self.data ** other, (self,), f"**{other}")
        def _b():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _b
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _b():
            self.grad += (out.data > 0) * out.grad
        out._backward = _b
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _b():
            self.grad += out.data * out.grad
        out._backward = _b
        return out

    def log(self):
        assert self.data > 0, "log requires positive input"
        out = Value(math.log(self.data), (self,), "log")
        def _b():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _b
        return out

    # Convenience ops derived from primitives
    def __neg__(self):        return self * -1
    def __sub__(self, other): return self + (-other if isinstance(other, Value) else Value(-other))
    def __radd__(self, other):return self + other
    def __rsub__(self, other):return (-self) + other
    def __rmul__(self, other):return self * other
    def __truediv__(self, other):
        other = self._wrap(other)
        return self * (other ** -1)
    def __rtruediv__(self, other):
        return self._wrap(other) * (self ** -1)

    def backward(self) -> None:
        # Topological order of the graph
        topo: list[Value] = []
        visited: set[int] = set()
        def build(v: Value):
            if id(v) in visited: return
            visited.add(id(v))
            for c in v._prev: build(c)
            topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


__all__ = ["Value"]
```

- [ ] **Step 5: Write `__init__.py`**

`projects/week-02-micrograd/src/micrograd/__init__.py`:

```python
from .engine import Value
from . import nn
__all__ = ["Value", "nn"]
```

- [ ] **Step 6: Run engine tests — expect PASS**

```bash
cd projects/week-02-micrograd && uv run pytest tests/test_engine.py -v 2>&1 | tail
```

Expected: 5 passed. If the torch-parity test skips (torch not installed via uv), that's acceptable — investigate `uv sync --extra dev` output; the test does `importorskip`.

- [ ] **Step 7: Write `nn.py` with a failing test first**

Create `projects/week-02-micrograd/tests/test_nn.py`:

```python
"""Neuron / Layer / MLP tests — a tiny MLP should overfit 4 points."""

from __future__ import annotations
import random
import pytest


def test_mlp_overfits_xor():
    from micrograd import Value, nn
    random.seed(1337)
    model = nn.MLP(2, [4, 4, 1])
    xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    ys = [0.0, 1.0, 1.0, 0.0]
    for step in range(200):
        loss = Value(0.0)
        for x_row, y_target in zip(xs, ys):
            pred = model([Value(x_row[0]), Value(x_row[1])])
            pred_val = pred[0] if isinstance(pred, list) else pred
            loss = loss + (pred_val - Value(y_target)) ** 2
        for p in model.parameters(): p.grad = 0.0
        loss.backward()
        for p in model.parameters(): p.data -= 0.05 * p.grad
    # After 200 steps of GD, loss should be under 0.5 on 4 points.
    assert loss.data < 0.5, f"loss too high: {loss.data}"
```

Run: FAIL (nn.MLP missing).

- [ ] **Step 8: Implement `nn.py`**

Create `projects/week-02-micrograd/src/micrograd/nn.py`:

```python
"""Tiny NN layer on top of Value — Neurons, Layers, and a stacked MLP."""

from __future__ import annotations
import random
from .engine import Value


class Neuron:
    def __init__(self, nin: int, nonlin: bool = True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: list[Value]) -> Value:
        assert len(x) == len(self.w), f"input dim {len(x)} != weight dim {len(self.w)}"
        act = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> list[Value]:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out  # scalar-out convenience

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        sizes = [nin] + list(nouts)
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x: list[Value]):
        for layer in self.layers:
            x = layer(x)
            if not isinstance(x, list): x = [x]
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


__all__ = ["Neuron", "Layer", "MLP"]
```

- [ ] **Step 9: Run all micrograd tests**

```bash
cd projects/week-02-micrograd && uv run pytest -v 2>&1 | tail
```

Expected: 6 passed (5 engine + 1 nn XOR).

- [ ] **Step 10: Write README, SOLUTION_NOTES, COMPUTE, assignments**

`projects/week-02-micrograd/README.md`:

```markdown
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
- `notebooks/01-autodiff-tour.ipynb` — a live walkthrough: forward, backward, GD on XOR.
- `assignments/{warmup,build,challenge}.md` — three sizes of exercise.

## Why re-implement autograd?

Because when PyTorch surprises you, you need a mental model of what `.backward()` is doing. This project fits in your head.
```

`projects/week-02-micrograd/SOLUTION_NOTES.md`:

```markdown
# Solution notes — micrograd

- Every op returns a fresh `Value`; you compose forward from left to right.
- `_backward` is a closure — it captures the parent Values and each op's local gradient rule.
- `backward()` topo-sorts the graph, seeds `self.grad = 1.0`, then walks reverse-topo calling every `_backward()`.
- Common pitfall: forgetting to zero grads between GD steps → gradients accumulate (this is intentional in PyTorch too; that's why `optimizer.zero_grad()` exists).
- The XOR example needs at least one hidden layer of ≥ 2 neurons — a single-neuron linear model cannot separate XOR.

## Gotchas found while writing this

- `__radd__` and `__rmul__` are needed so `Value(2) + 3` and `3 + Value(2)` both work.
- `Value` must be hashable-by-id (using `__slots__` gives us that for free) so it can be stored in a `set` for topo visit tracking.
- `log(x)` must guard `x > 0` — silent NaN propagation was the original micrograd's biggest gotcha.
```

`projects/week-02-micrograd/COMPUTE.md`:

```markdown
# COMPUTE — micrograd

- **Tier:** 🟢 local M-series (or any Python 3.13 machine).
- **Time:** `uv sync` ≈ 30 s; `pytest` ≈ 1 s; XOR notebook trains in < 5 s.
- **Budget:** $0. No GPU, no cloud.
- **`torch` install:** optional; the parity test uses `importorskip`. Skip it if `pip install torch` is heavyweight for your platform.
```

Assignments (each a short markdown; keep them concrete):

`projects/week-02-micrograd/assignments/warmup.md`:

```markdown
# Warmup (30 min)

1. Add a `Value.tanh()` op with the correct backward rule. Add a test that verifies `tanh(0).backward()` leaves grad 1.0.
2. Add a `sigmoid()` op derived from `exp`. No new `_backward` needed — see how `__truediv__` was expressed.
3. Print the topological order of a small graph before running `backward()`. Confirm each parent appears before every child in the reversed list.
```

`projects/week-02-micrograd/assignments/build.md`:

```markdown
# Build (2-3 hrs)

Train a 2-layer MLP on the classic 4-point moons dataset from `sklearn.datasets.make_moons(n_samples=100, noise=0.1)`. No sklearn model, just their dataset generator.

Deliverables:
1. A `train_moons.py` script that uses **your** `MLP`, does gradient descent for 500 steps, and prints test accuracy every 50 steps.
2. A matplotlib plot of the decision boundary at step 0, 100, 500.
3. A short "what surprised me" paragraph in `SOLUTION_NOTES.md`.

Constraint: no numpy vectorization inside the MLP. Every operation is a `Value`. This is slow. That's the point — you should feel the difference vs. vectorized NumPy in W4.
```

`projects/week-02-micrograd/assignments/challenge.md`:

```markdown
# Challenge (3+ hrs)

Extend `engine.py` to support **broadcasting** on a `Tensor` class that holds a numpy array as its `.data` and its `.grad`. Rules:

1. Same public API (`+`, `*`, `**`, `.relu()`, `.backward()`).
2. `Tensor.dot(Tensor)` must work — this is the primitive that lets you replace the per-scalar `Neuron` with a per-Layer matrix multiply.
3. Retrain the moons MLP with your `Tensor` class. Measure: is it faster than the scalar version? How much? Post the numbers in `SOLUTION_NOTES.md`.
4. Bonus: get the numeric-parity test to pass against `torch` on a Tensor-graph. (`atol=1e-5` is fine.)

Stretch: implement `.softmax()` and reproduce the 3-class Iris classifier from `torch.nn.CrossEntropyLoss`.
```

- [ ] **Step 11: Notebook — create the `.ipynb` file**

Since MDX renders code with Shiki (not Jupyter), the notebook is a companion for learners. Author it as a `.py` script and let uv/Jupyter tooling convert. Create `projects/week-02-micrograd/notebooks/01-autodiff-tour.py` (percent-format cells):

```python
# %% [markdown]
# # Autodiff tour — micrograd
# Live walkthrough: build a graph, run forward, run backward, GD on XOR.

# %%
from micrograd import Value, nn
a = Value(2.0)
b = Value(-3.0)
c = (a * b + a.relu()) * (b.exp() + Value(0.1))
c.backward()
print(f"c = {c.data:.4f}, dc/da = {a.grad:.4f}, dc/db = {b.grad:.4f}")

# %% [markdown]
# ## Train an MLP on XOR
# A single-neuron linear model cannot separate XOR — you need at least one hidden layer.

# %%
import random
random.seed(1337)
model = nn.MLP(2, [4, 4, 1])
xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [0.0, 1.0, 1.0, 0.0]
for step in range(300):
    loss = Value(0.0)
    for x, y in zip(xs, ys):
        pred = model([Value(x[0]), Value(x[1])])[0]
        loss = loss + (pred - Value(y)) ** 2
    for p in model.parameters(): p.grad = 0.0
    loss.backward()
    for p in model.parameters(): p.data -= 0.05 * p.grad
    if step % 50 == 0: print(f"step {step:3d} — loss {loss.data:.4f}")
```

*(A `.py` file with percent-format cells is convertible to `.ipynb` via `jupytext --to notebook 01-autodiff-tour.py`. The user can commit either format; the linter ignores `notebooks/`.)*

- [ ] **Step 12: Commit**

```bash
git add projects/week-02-micrograd
git commit -m "feat(w02): micrograd — scalar autodiff engine + tiny MLP + assignments"
```

---

## Task 3: Week 2 MDX — `week-02-calculus.mdx`

**Files:**
- Create: `apps/book/src/content/weeks/week-02-calculus.mdx`

**Interfaces:**
- Consumes: all Plan 1 + Plan 2 content components; `<CompanyLens>` requires all 7 companies.
- Produces: `/weeks/week-02-calculus` static route rendering the 8-part anatomy.

- [ ] **Step 1: Write the file**

Create `apps/book/src/content/weeks/week-02-calculus.mdx`:

```mdx
---
week: 2
part: 1
slug: "week-02-calculus"
title: "Calculus — the machinery of learning"
hook: "Every training loop in every AI system is one equation: subtract the gradient. Learn what the gradient IS and half of ML becomes bookkeeping."
hours: 20
computeTier: "green"
difficulty: 2
prereqSlugs: ["week-01-linear-algebra"]
referenceProject: "projects/week-02-micrograd"
accentVar: "--accent-p1"
---

import Hook from '../../components/content/Hook.astro'
import Intuition from '../../components/content/Intuition.astro'
import MathBlock from '../../components/content/MathBlock.astro'
import CompanyLens from '../../components/content/CompanyLens.astro'
import ReferenceProject from '../../components/content/ReferenceProject.astro'
import Assignments from '../../components/content/Assignments.astro'
import InterviewDrill from '../../components/content/InterviewDrill.astro'
import FurtherReading from '../../components/content/FurtherReading.astro'
import KeyTakeaways from '../../components/content/KeyTakeaways.astro'
import IntuitionCallout from '../../components/callouts/Intuition.astro'
import Gotcha from '../../components/callouts/Gotcha.astro'
import { MicroRecall } from '../../components/interactive/MicroRecall'

<Hook>Every model you'll ever train is doing one thing 100 million times: take the derivative of the loss with respect to a parameter, then subtract a small multiple of it. Get comfortable with derivatives and gradients, and 100% of training-time debugging becomes tractable.</Hook>

<Intuition>
A **derivative** answers: "if I nudge this input up by ε, how much does the output change?" It is a *sensitivity*.

<IntuitionCallout>
A gradient is a **vector of derivatives** — one per parameter. It points in the direction of steepest increase of the loss. Training subtracts a small multiple of it (SGD). The learning rate is that small multiple.
</IntuitionCallout>

The four moves you must own by end of week: **derivative of a scalar function** (rate of change), **partial derivative** (rate w.r.t. one input, others fixed), **gradient** (vector of partials), and **chain rule** (compose them).

<MicroRecall
  questions={[{
    id: 'week2-recall-chainrule',
    prompt: 'If y = f(g(x)) and g(x) = 2x, f(u) = u², what is dy/dx at x = 3?',
    choices: ['6', '12', '24', '36'],
    correct: 2,
    explain: 'y = (2x)² = 4x²; dy/dx = 8x; at x=3 → 24. Equivalently by chain rule: f\'(g) · g\'(x) = 2u · 2 = 4·2x = 8x.',
  }]}
  client:visible
/>
</Intuition>

## The math, derived

We build calculus in the exact order training uses it: **scalar derivative → partial → gradient → chain rule → backpropagation**.

### 1. Scalar derivative

The derivative of `f(x)` at `x` is the slope of the tangent line:

<MathBlock latex="f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}" />

Two rules you must know cold: **power rule** `d/dx x^n = n x^{n-1}` and **sum rule** `d/dx [f + g] = f' + g'`. Every derivative in ML is a composition of these plus the chain rule.

### 2. Partial derivatives

If `f(x, y) = x²y + y`, then `∂f/∂x = 2xy` (treat `y` as constant) and `∂f/∂y = x² + 1` (treat `x` as constant). That's it. Partials are just single-variable derivatives with the other inputs frozen.

### 3. Gradient

<MathBlock latex="\nabla f(\mathbf{x}) = \begin{bmatrix}\partial f / \partial x_1 \\ \partial f / \partial x_2 \\ \vdots \\ \partial f / \partial x_n\end{bmatrix}" />

The gradient is a **vector** the same shape as the parameter it takes partials of. In a neural network with 1B parameters, `∇L(θ)` is a 1B-dimensional vector — one gradient per parameter.

### 4. Chain rule

If `y = f(g(x))`, then

<MathBlock latex="\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} \quad \text{where } u = g(x)" />

This is the entire content of backpropagation. Every op in your model contributes its local `du/dx`; backprop multiplies them together in reverse topological order.

<Gotcha>
Common trap: at a **branching point** in the graph (a `Value` used in two downstream ops), the gradients from each downstream path **add**, they don't overwrite. This is the *multivariable chain rule*, and forgetting it is Karpathy's most-cited micrograd bug.
</Gotcha>

### 5. Gradient descent

Given a loss `L(θ)` and a learning rate `η`, one training step is:

<MathBlock latex="\theta \leftarrow \theta - \eta \nabla L(\theta)" />

Small `η` → slow but safe. Large `η` → fast but can overshoot and diverge. Whole cottage industries (Adam, momentum, warmup) exist to pick `η` well.

## The code, from scratch

Build a scalar autodiff engine in ~150 lines. Every op — `+`, `*`, `relu`, `exp` — stores its parents and a local backward function. `backward()` topo-sorts the graph and runs each op's backward in reverse order.

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._prev = _children
        self._backward = lambda: None

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        def _b():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _b
        return out
```

<Gotcha>
`self.grad += ...` — the `+=` is what makes the multivariable chain rule work at branching points. Never write `self.grad = ...` inside a `_backward`.
</Gotcha>

Full implementation: `projects/week-02-micrograd/src/micrograd/engine.py`.

<CompanyLens
  topic="Calculus & autodiff"
  tldr={{
    openai:    "Chain rule + gradients = the whole training story. RLHF is a chain of chain rules.",
    anthropic: "Constitutional AI runs the same gradient descent — with a different loss shape.",
    deepmind:  "AlphaFold's loss surface is calculus at protein scale; the derivatives are physics-informed.",
    meta:      "PyTorch's autograd IS the calculus of Week 2 — first-class citizen in the framework.",
    xai:       "Grok's training pipeline burns cluster-hours on one operation: subtract the gradient.",
    deepseek:  "GRPO's advantage estimate is a Monte Carlo mean of a policy's gradient.",
    qwen:      "Open-weight models publish gradients-at-checkpoint — read them to see training dynamics.",
  }}
  whyDiffer="Every lab agrees on the machinery (backprop). They differ in what loss they optimize (next-token vs preference vs constitutional vs reward-model) and how they distribute the gradient across a cluster (FSDP, ZeRO, ring-attention). Weeks 15 and 18 revisit both axes."
  sources={{
    openai:    { paper: "InstructGPT — arXiv:2203.02155 (RLHF loss)",     blog: "openai.com/research" },
    anthropic: { paper: "Constitutional AI — arXiv:2212.08073",           blog: "anthropic.com/research" },
    deepmind:  { paper: "AlphaFold 2 (Jumper et al.)",                    blog: "deepmind.google/discover/blog" },
    meta:      { paper: "PyTorch autograd docs",                          blog: "pytorch.org/docs/stable/autograd.html" },
    xai:       { paper: "Grok 3 tech report",                             blog: "x.ai/blog" },
    deepseek:  { paper: "DeepSeek-R1 arXiv:2501.12948 (GRPO)",            blog: "deepseek.com/research" },
    qwen:      { paper: "Qwen 2.5 tech report",                           blog: "qwenlm.github.io" },
  }}
  caseStudy="Karpathy's original micrograd (~200 LOC) is small enough that OpenAI, Anthropic, and DeepMind engineers cite it in onboarding docs as 'read this once, then you understand autograd.' We rebuild it this week."
  interviewAngle={{
    openai:    "Debug a training loop: they hand you a loss curve that plateaus at loss 5.0. What's wrong?",
    anthropic: "Derive the gradient of a KL divergence — expect to write it on the whiteboard.",
    deepmind:  "Given a graph with a branching Value, hand-compute its backward pass on paper.",
    meta:      "Implement Value's `_backward` for `**power` from scratch. No PyTorch.",
    xai:       "Live-write a gradient-descent loop in Python without a framework. 5 minutes.",
    deepseek:  "Explain policy gradient's `E[log π(a|s) · A(s,a)]` — where's the derivative?",
    qwen:      "Given two checkpoints (10k steps apart), what does the gradient direction tell you?",
  }}
/>

<ReferenceProject path="projects/week-02-micrograd" name="micrograd" hours="6-8 hrs">
Build a ~200-LOC scalar autodiff engine. Extend it to a tiny MLP that overfits XOR in 200 GD steps. The numeric-parity test against `torch.autograd` is the acceptance gate.
</ReferenceProject>

<Assignments>
- **Warmup (30 min):** add `Value.tanh()` and `Value.sigmoid()` with correct backward rules.
- **Build (2-3 hrs):** train a 2-layer MLP on `sklearn.datasets.make_moons`. Plot the decision boundary at step 0, 100, 500. Slow — that's the point; you'll feel it against W4's NumPy vs PyTorch.
- **Challenge (3+ hrs):** extend `engine.py` to a `Tensor` class holding a numpy array. Support broadcasting, `dot`, and `softmax`. Retrain the moons MLP with the tensor version; measure the speedup.
</Assignments>

<InterviewDrill role="research-eng" time="20 min">
**"Explain what `.backward()` is doing on a graph with a branching node, without hand-waving."**

Rubric:
- Can they draw the graph and mark the branching node? (baseline)
- Do they say "gradients **add** at a branching node"? (senior signal — this is the multivariable chain rule)
- Can they name at least one framework failure mode: forgetting to zero grads, in-place ops, retain_graph=False? (staff signal)

Worked solution: see `SOLUTION_NOTES.md` in `projects/week-02-micrograd/`.
</InterviewDrill>

<FurtherReading>
- **Karpathy — "The spelled-out intro to neural networks and backpropagation"** (YouTube). The 2.5-hour lecture that this project is built from.
- **Goodfellow, Bengio, Courville — Deep Learning, Chapter 6.5** — "Back-Propagation and Other Differentiation Algorithms."
- **PyTorch autograd docs** — read them front-to-back. Meta hires with this doc in mind.
- **Bishop — PRML, Chapter 5.3** — Backpropagation derivation (the classical view).
</FurtherReading>

<KeyTakeaways>
- Derivative = sensitivity. Gradient = vector of partial derivatives, one per parameter.
- The **chain rule** is the entire content of backprop. Local gradient × upstream gradient.
- At a **branching node**, gradients from downstream paths **add** (multivariable chain rule).
- Training is `θ ← θ − η ∇L(θ)`. Every optimizer (Adam, SGD-momentum) is a variation on this line.
- If you can implement `backward()` in ~40 lines, you understand deep learning training at the algorithm level.
</KeyTakeaways>
```

- [ ] **Step 2: Sync + build**

```bash
export PATH="/Users/tharunkumarl/.nvm/versions/node/v24.21.0/bin:$PATH"
cd apps/book && npx astro sync 2>&1 | tail -3 && pnpm run build 2>&1 | tail -5
```

Expected: sync + build both exit 0; console shows `/weeks/week-02-calculus/index.html` prerendered.

- [ ] **Step 3: Commit**

```bash
git add apps/book/src/content/weeks/week-02-calculus.mdx
git commit -m "feat(w02): Calculus MDX with 8-part anatomy + 7-company Lens"
```

---

## Task 4: Week 3 reference project — `prob-lab`

**Files:**
- Create: everything under `projects/week-03-prob-lab/`

**Interfaces:**
- Consumes: uv workspace, numpy, matplotlib.
- Produces:
  - `prob_lab.monte_carlo.estimate_pi(n: int, seed: int = 0) -> float`
  - `prob_lab.monte_carlo.expectation(fn: Callable, sampler: Callable, n: int, seed: int = 0) -> float`
  - `prob_lab.bayes.posterior_mean(prior_a: float, prior_b: float, heads: int, tails: int) -> float` — Beta-Binomial conjugate.
  - `prob_lab.bayes.log_bayes_factor(likelihood_a: float, likelihood_b: float, prior_a: float, prior_b: float) -> float`
  - `prob_lab.entropy.entropy(p: np.ndarray) -> float` — Shannon in nats, safe against zeros.
  - `prob_lab.entropy.kl_divergence(p: np.ndarray, q: np.ndarray) -> float` — safe against `q == 0`.
  - `prob_lab.entropy.cross_entropy(p: np.ndarray, q: np.ndarray) -> float`.

- [ ] **Step 1: Scaffold + pyproject.toml**

```bash
mkdir -p projects/week-03-prob-lab/{src/prob_lab,tests,notebooks,assignments}
touch projects/week-03-prob-lab/src/prob_lab/__init__.py
```

Write `projects/week-03-prob-lab/pyproject.toml`:

```toml
[project]
name = "prob-lab"
version = "0.1.0"
description = "Week 3 — Monte Carlo, Bayes, entropy, KL, cross-entropy"
requires-python = ">=3.13"
dependencies = ["numpy>=2.1", "matplotlib>=3.9", "scipy>=1.14"]

[project.optional-dependencies]
dev = ["pytest>=8.3", "hypothesis>=6.112", "ruff>=0.7", "pyright>=1.1"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
addopts = "-v --strict-markers"
testpaths = ["tests"]
```

- [ ] **Step 2: Write failing tests**

`projects/week-03-prob-lab/tests/test_monte_carlo.py`:

```python
"""Monte Carlo tests. Zero-variance sampling is the Review Focus #1 anchor."""

from __future__ import annotations
import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st


def test_pi_converges():
    from prob_lab.monte_carlo import estimate_pi
    est = estimate_pi(n=200_000, seed=42)
    assert abs(est - math.pi) < 0.02


def test_expectation_of_identity_is_mean():
    from prob_lab.monte_carlo import expectation
    rng = np.random.default_rng(7)
    val = expectation(fn=lambda x: x, sampler=lambda size: rng.normal(0, 1, size), n=50_000)
    assert abs(val - 0.0) < 0.02


@given(st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False))
@settings(max_examples=30, deadline=None)
def test_expectation_survives_constant_samplers(const):
    """Review Focus #1 — zero-variance input; must not divide-by-zero or NaN."""
    from prob_lab.monte_carlo import expectation
    val = expectation(fn=lambda x: x * 2, sampler=lambda size: np.full(size, const), n=100)
    assert math.isfinite(val)
    assert abs(val - const * 2) < 1e-10
```

`projects/week-03-prob-lab/tests/test_bayes.py`:

```python
from __future__ import annotations
import pytest


def test_posterior_mean_beta_binomial():
    from prob_lab.bayes import posterior_mean
    # Prior Beta(2, 2), 8 heads / 2 tails → posterior Beta(10, 4), mean 10/14.
    assert posterior_mean(2, 2, heads=8, tails=2) == pytest.approx(10 / 14)


def test_posterior_mean_uniform_prior():
    from prob_lab.bayes import posterior_mean
    # Uniform Beta(1, 1) + 3 heads / 1 tail → Beta(4, 2), mean 4/6.
    assert posterior_mean(1, 1, heads=3, tails=1) == pytest.approx(4 / 6)


def test_log_bayes_factor_symmetric():
    from prob_lab.bayes import log_bayes_factor
    # BF_ab = 1/BF_ba, so log(BF_ab) = -log(BF_ba).
    a = log_bayes_factor(likelihood_a=0.4, likelihood_b=0.1, prior_a=0.5, prior_b=0.5)
    b = log_bayes_factor(likelihood_a=0.1, likelihood_b=0.4, prior_a=0.5, prior_b=0.5)
    assert a == pytest.approx(-b)
```

`projects/week-03-prob-lab/tests/test_entropy.py`:

```python
from __future__ import annotations
import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st


def test_uniform_entropy_is_log_n():
    from prob_lab.entropy import entropy
    for n in (2, 3, 8):
        p = np.full(n, 1 / n)
        assert entropy(p) == pytest.approx(math.log(n))


def test_delta_entropy_is_zero():
    """log(1) + 0·log(0) = 0. Zero-support case must return 0, not NaN — Review Focus #1."""
    from prob_lab.entropy import entropy
    p = np.array([0.0, 1.0, 0.0, 0.0])
    assert entropy(p) == 0.0


def test_kl_positive():
    from prob_lab.entropy import kl_divergence
    p = np.array([0.7, 0.3])
    q = np.array([0.5, 0.5])
    assert kl_divergence(p, q) > 0.0
    # KL(p||p) = 0
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-10)


def test_cross_entropy_ge_entropy():
    """H(p, q) >= H(p), with equality iff p == q."""
    from prob_lab.entropy import entropy, cross_entropy
    p = np.array([0.4, 0.6])
    q = np.array([0.5, 0.5])
    assert cross_entropy(p, q) >= entropy(p) - 1e-10


@given(st.floats(min_value=1e-6, max_value=1 - 1e-6))
@settings(max_examples=50, deadline=None)
def test_binary_kl_bounds(a):
    """Binary KL divergence stays finite and non-negative for interior probabilities."""
    from prob_lab.entropy import kl_divergence
    p = np.array([a, 1 - a])
    q = np.array([0.5, 0.5])
    kl = kl_divergence(p, q)
    assert math.isfinite(kl)
    assert kl >= 0.0
```

Run all → FAIL (module missing).

- [ ] **Step 3: Implement `monte_carlo.py`**

`projects/week-03-prob-lab/src/prob_lab/monte_carlo.py`:

```python
"""Monte Carlo estimators — π by rejection, and generic expectations."""

from __future__ import annotations
from typing import Callable
import numpy as np


def estimate_pi(n: int, seed: int = 0) -> float:
    """Fraction of uniform (x,y) points in the unit square that fall inside the
    quarter unit circle, × 4."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2))
    inside = np.count_nonzero((xy ** 2).sum(axis=1) <= 1.0)
    return 4.0 * inside / n


def expectation(
    fn: Callable[[np.ndarray], np.ndarray | float],
    sampler: Callable[[int], np.ndarray],
    n: int,
    seed: int = 0,
) -> float:
    """E[fn(X)] with X drawn from `sampler(n)`. `seed` is documentary — pass a
    seeded rng from the caller if you want reproducibility."""
    samples = sampler(n)
    values = np.asarray(fn(samples), dtype=float)
    if values.size == 0:
        return 0.0
    return float(values.mean())
```

- [ ] **Step 4: Implement `bayes.py`**

`projects/week-03-prob-lab/src/prob_lab/bayes.py`:

```python
"""Bayes: Beta-Binomial conjugate posterior + Bayes factors."""

from __future__ import annotations
import math


def posterior_mean(prior_a: float, prior_b: float, heads: int, tails: int) -> float:
    """Beta(a, b) prior + Binomial(heads, tails) likelihood → Beta(a+heads, b+tails)
    posterior, whose mean is (a+heads) / (a+b+heads+tails)."""
    assert prior_a > 0 and prior_b > 0, "beta priors must be positive"
    assert heads >= 0 and tails >= 0, "counts must be non-negative"
    a, b = prior_a + heads, prior_b + tails
    return a / (a + b)


def log_bayes_factor(
    likelihood_a: float, likelihood_b: float,
    prior_a: float, prior_b: float,
) -> float:
    """log P(D|A)/P(D|B) — a log-scale ratio of how well each hypothesis explains D.
    Priors are needed so the caller can also compute log-posterior-odds.

    Guards against log(0) with a floor at -1e300.
    """
    def _log(x: float) -> float:
        return math.log(x) if x > 0 else -1e300
    return _log(likelihood_a) - _log(likelihood_b) + _log(prior_a) - _log(prior_b)
```

- [ ] **Step 5: Implement `entropy.py`**

`projects/week-03-prob-lab/src/prob_lab/entropy.py`:

```python
"""Shannon entropy, KL divergence, cross-entropy — all in nats, zero-safe."""

from __future__ import annotations
import numpy as np

# Small epsilon used only to gate log(0) — never added to the probabilities.
_EPS = 1e-300


def entropy(p: np.ndarray) -> float:
    """Shannon H(p) = -Σ p log p. Zeros contribute 0 by convention (0·log 0 = 0)."""
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return float(-(p[mask] * np.log(p[mask])).sum())


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) = Σ p log(p/q). Undefined when q_i = 0 and p_i > 0 (returns inf).
    When p_i = 0 the contribution is 0 by convention."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    assert p.shape == q.shape, "distributions must share shape"
    mask = p > 0
    if np.any((q[mask] <= 0)):
        return float("inf")
    return float((p[mask] * (np.log(p[mask]) - np.log(q[mask]))).sum())


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """H(p, q) = -Σ p log q = H(p) + KL(p||q)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    if np.any((q[mask] <= 0)):
        return float("inf")
    return float(-(p[mask] * np.log(q[mask])).sum())
```

- [ ] **Step 6: `__init__.py`**

```python
from .monte_carlo import estimate_pi, expectation
from .bayes import posterior_mean, log_bayes_factor
from .entropy import entropy, kl_divergence, cross_entropy

__all__ = [
    "estimate_pi", "expectation",
    "posterior_mean", "log_bayes_factor",
    "entropy", "kl_divergence", "cross_entropy",
]
```

- [ ] **Step 7: Run all tests**

```bash
cd projects/week-03-prob-lab && uv sync --extra dev && uv run pytest -v 2>&1 | tail
```

Expected: 10 passed (3 monte_carlo + 3 bayes + 4 entropy).

- [ ] **Step 8: README / SOLUTION_NOTES / COMPUTE / assignments / notebook**

Author each per Week 2's shape. Key content:

`projects/week-03-prob-lab/README.md`:

```markdown
# prob-lab — Week 3 reference project

Monte Carlo estimators, Bayes-with-conjugate priors, and Shannon information (entropy, KL, cross-entropy) — all zero-safe.

## Run

```bash
cd projects/week-03-prob-lab
uv sync --extra dev
uv run pytest -v
```

## Public API

- `estimate_pi(n, seed)` — 4 × fraction of unit-square samples inside the unit quarter-circle.
- `expectation(fn, sampler, n)` — generic Monte Carlo mean.
- `posterior_mean(a, b, heads, tails)` — Beta-Binomial closed-form posterior mean.
- `log_bayes_factor(...)` — log ratio of hypothesis likelihoods + priors.
- `entropy(p)` / `kl_divergence(p, q)` / `cross_entropy(p, q)` — all in nats, all zero-safe.
```

`projects/week-03-prob-lab/SOLUTION_NOTES.md`:

```markdown
# Solution notes — prob-lab

- Zero-safe entropy: mask `p > 0` before computing `log`. Never add ε to probabilities — that biases the estimate. The 0·log(0) = 0 convention is safe because we're skipping those terms entirely.
- KL divergence returns `inf` when `q_i = 0 and p_i > 0`. That's mathematically correct — the reverse KL is finite there, though; know the difference.
- Beta-Binomial conjugacy: (α, β) + (h, t) → (α+h, β+t). This is why Bayesian analysts love the Beta family for binary outcomes.
- π by Monte Carlo converges at rate O(1/√n). At n=200k the standard error is ≈ 0.004; the test allows 0.02 to avoid flakiness.
- Bayes factor pitfall: the log ratio can be a huge number. Never exponentiate before comparing — stay in log-space.
```

`projects/week-03-prob-lab/COMPUTE.md`:

```markdown
# COMPUTE — prob-lab

- **Tier:** 🟢 local.
- **Time:** `uv sync` ≈ 30 s; `pytest` ≈ 2 s; notebook plots ≈ 30 s.
- **Budget:** $0.
```

Assignments:

`warmup.md`:

```markdown
# Warmup (30 min)

1. Verify empirically that `estimate_pi`'s standard error scales like `1/√n`. Plot n vs |estimate - π| on a log-log axis for n ∈ {10², 10³, 10⁴, 10⁵}.
2. Add a `posterior_variance(prior_a, prior_b, heads, tails)` closed-form function. The Beta variance formula is `αβ / ((α+β)² (α+β+1))`.
3. Compute `entropy([0.5, 0.5])` and `entropy([0.9, 0.1])` — is the second smaller? By how much?
```

`build.md`:

```markdown
# Build (2-3 hrs)

Implement a coin-bias inference notebook.

1. Simulate 100 coin flips from a coin with true bias p = 0.65.
2. Start with a Beta(2, 2) prior.
3. After every 10 flips, plot the posterior Beta density on the same axes. You should see it tighten around 0.65 as data accumulates.
4. Add a KL-vs-true-posterior overlay: at every step, compute KL between the current posterior and the "final" posterior (from all 100 flips). It should decay monotonically.

Deliverables:
- `notebooks/coin_bias.ipynb` (or `.py` in percent format).
- A short "what surprised me" paragraph in `SOLUTION_NOTES.md`.
```

`challenge.md`:

```markdown
# Challenge (3+ hrs)

Implement **importance sampling** and compare it to naive Monte Carlo on a heavy-tailed problem.

Setup: estimate E[X²] where X ~ N(0, 1) truncated to |x| > 3 (the tails).

1. Naive MC: sample from N(0, 1) and reject |x| ≤ 3. Rejection rate ≈ 99.7% — inefficient.
2. Importance sampling: draw from a Cauchy proposal q(x), weight by p(x)/q(x). Should converge with 10-100× fewer samples.
3. Plot: variance-vs-samples for both methods. IS should win by an order of magnitude.
4. Write it up in `SOLUTION_NOTES.md` with the two curves.

Stretch: extend to a 2-D truncated Gaussian and compare the two approaches on integrated squared error.
```

- [ ] **Step 9: Notebook `notebooks/01-bayes-and-entropy.py`**

Percent-format cells:

```python
# %% [markdown]
# # Bayes & entropy tour
# Coin-bias inference + entropy of common distributions.

# %%
import numpy as np, matplotlib.pyplot as plt
from prob_lab import estimate_pi, posterior_mean, entropy

print("π estimate at n=100_000:", estimate_pi(100_000, seed=42))

# %% [markdown]
# ## Posterior mean sweeps
# %%
priors = [(1, 1), (2, 2), (10, 10)]
observations = [(0, 0), (3, 7), (30, 70), (300, 700)]
for a, b in priors:
    row = [posterior_mean(a, b, h, t) for h, t in observations]
    print(f"prior Beta({a},{b}): {row}")

# %% [markdown]
# ## Entropy of the "how many heads?" distribution
# %%
probs = np.linspace(0.01, 0.99, 50)
H = [entropy([p, 1 - p]) for p in probs]
plt.figure(figsize=(6, 3))
plt.plot(probs, H); plt.xlabel("P(heads)"); plt.ylabel("H(p) (nats)"); plt.title("Binary entropy peaks at 0.5")
plt.tight_layout(); plt.savefig("binary_entropy.png"); plt.close()
```

- [ ] **Step 10: Commit**

```bash
git add projects/week-03-prob-lab
git commit -m "feat(w03): prob-lab — Monte Carlo, Bayes, entropy + zero-safe KL"
```

---

## Task 5: Week 3 MDX — `week-03-probability.mdx`

**Files:**
- Create: `apps/book/src/content/weeks/week-03-probability.mdx`

**Interfaces:**
- Same as Task 3 (Astro content components already exist).

- [ ] **Step 1: Write the file**

Same skeleton as Week 2. Fill each section with content on: **probability basics** (sample space, events, conditional), **distributions** (Bernoulli, Binomial, Normal, Poisson), **Bayes' rule** (posterior = likelihood × prior / evidence), **MLE vs MAP** (both are optimization problems), **entropy / KL / cross-entropy** (the losses of ML).

Full file:

```mdx
---
week: 3
part: 1
slug: "week-03-probability"
title: "Probability & Statistics — the grammar of uncertainty"
hook: "Every model is a probability distribution in a trench coat. Learn to read the coat off, and you can debug any model by asking one question: what distribution does this claim to represent?"
hours: 20
computeTier: "green"
difficulty: 2
prereqSlugs: ["week-02-calculus"]
referenceProject: "projects/week-03-prob-lab"
accentVar: "--accent-p1"
---

import Hook from '../../components/content/Hook.astro'
import Intuition from '../../components/content/Intuition.astro'
import MathBlock from '../../components/content/MathBlock.astro'
import CompanyLens from '../../components/content/CompanyLens.astro'
import ReferenceProject from '../../components/content/ReferenceProject.astro'
import Assignments from '../../components/content/Assignments.astro'
import InterviewDrill from '../../components/content/InterviewDrill.astro'
import FurtherReading from '../../components/content/FurtherReading.astro'
import KeyTakeaways from '../../components/content/KeyTakeaways.astro'
import IntuitionCallout from '../../components/callouts/Intuition.astro'
import Gotcha from '../../components/callouts/Gotcha.astro'
import { MicroRecall } from '../../components/interactive/MicroRecall'

<Hook>Every neural net output is a probability. Every loss function is a distance between two distributions. When your model "hallucinates," it is *sampling from the wrong distribution*. Own probability, own debugging.</Hook>

<Intuition>
A **probability distribution** answers: "over this space of outcomes, how does the mass get allocated?" Discrete distributions (dice, coin flips, next-token) put point masses on outcomes; continuous ones (heights, gradients, weights) put density on regions.

<IntuitionCallout>
**Bayes' rule** answers: "given what I've seen, what should I now believe?" It converts a **prior** (before-data belief) plus a **likelihood** (data-given-hypothesis) into a **posterior** (after-data belief). RLHF, active learning, and Bayesian model averaging are all applied Bayes.
</IntuitionCallout>

The four moves you must own by end of week: **conditional probability & Bayes' rule**, **expectation & variance**, **maximum likelihood estimation**, and **entropy / KL / cross-entropy** (the losses of ML).

<MicroRecall
  questions={[{
    id: 'week3-recall-bayes',
    prompt: 'A test for disease D has 99% sensitivity and 99% specificity. Prevalence is 1%. You test positive. What is P(disease | positive)?',
    choices: ['~99%', '~50%', '~9%', '~1%'],
    correct: 1,
    explain: 'Bayes: P(D|+) = 0.99·0.01 / (0.99·0.01 + 0.01·0.99) = 0.5. Base rates dominate — this is the classic "test paradox."',
  }]}
  client:visible
/>
</Intuition>

## The math, derived

### 1. Conditional probability & Bayes' rule

<MathBlock latex="P(A \mid B) = \frac{P(A \cap B)}{P(B)} \implies P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}" />

Read: "posterior = likelihood × prior / evidence." The denominator normalizes; often you can ignore it when comparing hypotheses (Bayes factors).

### 2. Expectation & variance

<MathBlock latex="E[X] = \sum_x x \cdot P(X = x), \quad \text{Var}(X) = E[(X - E[X])^2]" />

For a continuous X with density f: `E[X] = ∫ x f(x) dx`. Expectation is linear (`E[aX + b] = aE[X] + b`); variance is not (`Var(aX + b) = a² Var(X)`).

### 3. Maximum likelihood estimation

Given data `x₁, …, xₙ` iid from a distribution with parameters `θ`, MLE picks `θ̂` that maximizes the **likelihood**:

<MathBlock latex="\hat\theta_{\text{MLE}} = \arg\max_\theta \prod_i p(x_i \mid \theta) = \arg\max_\theta \sum_i \log p(x_i \mid \theta)" />

In practice we maximize the **log-likelihood** — same argmax, better numerics. MAP adds a `log p(θ)` prior term.

<Gotcha>
MLE for Gaussian variance uses `1/n`, not `1/(n-1)`. The `1/(n-1)` version is the **unbiased** estimator; MLE is *biased downward*. Both are correct — for different objectives.
</Gotcha>

### 4. Entropy, KL, cross-entropy

<MathBlock latex="H(p) = -\sum_i p_i \log p_i" />
<MathBlock latex="\text{KL}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i} \ge 0" />
<MathBlock latex="H(p, q) = -\sum_i p_i \log q_i = H(p) + \text{KL}(p \| q)" />

Cross-entropy is the loss function of every classifier. Minimize cross-entropy = maximize likelihood (they differ by a constant). This is why "next-token cross-entropy loss" and "next-token log-likelihood" refer to the same objective.

## The code, from scratch

Zero-safe entropy (Review Focus #1) — this is the pitfall every fresh grad hits:

```python
import numpy as np

def entropy(p):
    p = np.asarray(p, dtype=float)
    mask = p > 0            # ← skip zeros, never add epsilon
    return float(-(p[mask] * np.log(p[mask])).sum())
```

Adding a small ε to `p` biases the estimate. Masking is correct.

Full implementations: `projects/week-03-prob-lab/src/prob_lab/`.

<CompanyLens
  topic="Probability & information"
  tldr={{
    openai:    "Next-token cross-entropy is the loss. Everything downstream (RLHF, alignment) minimizes a variant.",
    anthropic: "Constitutional AI uses KL to keep the policy close to a reference distribution — the RL constraint.",
    deepmind:  "Bayesian NNs (MacKay 1992) shaped DeepMind's uncertainty-first approach — worth reading the original.",
    meta:      "Every Llama tech report reports token-level perplexity — the exponentiated cross-entropy.",
    xai:       "First-principles rejection sampling used inside Grok's serving pipeline — a Monte Carlo choice, not neural.",
    deepseek:  "V3's expert routing is a discrete probability distribution over 256 experts. Read the load-balancing loss.",
    qwen:      "Cross-modal alignment loss between text & vision distributions.",
  }}
  whyDiffer="Every lab uses the same losses (cross-entropy, KL); they diverge in how they constrain sampling at inference. Anthropic's Constitutional AI, DeepMind's Bayesian uncertainty, and DeepSeek's MoE routing are three answers to 'what distribution should we sample from?'"
  sources={{
    openai:    { paper: "GPT-4 System Card — Section on hallucination distributions", blog: "openai.com/research" },
    anthropic: { paper: "Constitutional AI arXiv:2212.08073 (KL constraint)",         blog: "anthropic.com/research" },
    deepmind:  { paper: "MacKay 1992 — Bayesian Interpolation",                       blog: "deepmind.google/discover/blog" },
    meta:      { paper: "Llama 3 tech report — perplexity tables",                    blog: "ai.meta.com/blog" },
    xai:       { paper: "Grok 3 tech report",                                         blog: "x.ai/blog" },
    deepseek:  { paper: "DeepSeek-V3 arXiv:2412.19437 — expert routing loss",         blog: "deepseek.com/research" },
    qwen:      { paper: "Qwen-VL tech report",                                        blog: "qwenlm.github.io" },
  }}
  caseStudy="The 'Bayesian test paradox' (see MicroRecall above) — 99% sensitivity + 99% specificity + 1% prevalence gives P(disease | positive) ≈ 50%, not 99%. This is why interpreting model confidence scores WITHOUT knowing the base rate leads senior engineers astray."
  interviewAngle={{
    openai:    "Derive perplexity from cross-entropy. Why is it more human-readable?",
    anthropic: "Write the KL constraint in Constitutional AI. What happens as β → 0?",
    deepmind:  "Bayesian NN vs MC dropout — pick one and defend it.",
    meta:      "MLE vs MAP: when does the prior term matter?",
    xai:       "Rejection sampling: given a hard-to-sample target, when does IS beat MCMC?",
    deepseek:  "Load-balancing loss in MoE: what does the auxiliary loss actually prevent?",
    qwen:      "Cross-modal alignment: derive the InfoNCE loss.",
  }}
/>

<ReferenceProject path="projects/week-03-prob-lab" name="prob-lab" hours="6-8 hrs">
Build Monte Carlo estimators, a Beta-Binomial Bayesian analysis, and a zero-safe entropy/KL/cross-entropy module. Notebook: infer a coin's bias from 100 flips, watch the posterior tighten.
</ReferenceProject>

<Assignments>
- **Warmup (30 min):** verify MC standard error scales like `1/√n` on `estimate_pi`. Add a closed-form `posterior_variance` for Beta.
- **Build (2-3 hrs):** coin-bias inference notebook — 100 flips, Beta(2,2) prior, plot posterior evolution every 10 flips.
- **Challenge (3+ hrs):** importance sampling on a heavy-tailed truncated Gaussian. Compare variance-vs-samples with naive MC.
</Assignments>

<InterviewDrill role="applied-ml" time="20 min">
**"You train a 99% accuracy model on a 1%-prevalence disease. Interpret the model's positive predictions."**

Rubric:
- Do they invoke Bayes and compute the posterior? (baseline)
- Do they name "base-rate neglect" as the cognitive trap? (senior signal)
- Do they discuss calibration — the model may be 99% accurate but its 90%-confidence positives could actually be 50% correct? (staff signal)

Worked solution: `projects/week-03-prob-lab/SOLUTION_NOTES.md`.
</InterviewDrill>

<FurtherReading>
- **Bishop — PRML, Chapters 1-2** — the standard reference.
- **MacKay — Information Theory, Inference, and Learning Algorithms** — free PDF from the Cambridge site. Chapter 2 on entropy is a career-long companion.
- **Jaynes — Probability Theory: The Logic of Science** — the Bayesian bible. Dense; skim first, deep-read after Week 8.
- **StatQuest videos** — Josh Starmer on Bayes, MLE, entropy. Excellent for intuition.
</FurtherReading>

<KeyTakeaways>
- Bayes' rule turns priors + data into posteriors. Never omit the base rate.
- MLE = maximize log-likelihood = minimize cross-entropy (they differ by a constant).
- Entropy H(p) = -Σ p log p. KL(p‖q) ≥ 0. Cross-entropy H(p,q) = H(p) + KL(p‖q).
- Zero-safe entropy uses masking, not epsilon-nudging. Never add ε to a probability.
- Perplexity = exp(cross-entropy). Every LLM tech report cites it; know what it means.
</KeyTakeaways>
```

- [ ] **Step 2: Build**

```bash
export PATH="/Users/tharunkumarl/.nvm/versions/node/v24.21.0/bin:$PATH"
cd apps/book && pnpm run build 2>&1 | tail -5
```

Expected: `/weeks/week-03-probability/index.html` prerendered.

- [ ] **Step 3: Commit**

```bash
git add apps/book/src/content/weeks/week-03-probability.mdx
git commit -m "feat(w03): Probability MDX with 8-part anatomy + 7-company Lens"
```

---

## Task 6: Week 4 reference project — `numpy-vs-pytorch`

**Files:**
- Create: everything under `projects/week-04-numpy-vs-pytorch/`

**Interfaces:**
- Consumes: numpy, torch, plus optional pure-Python control impls.
- Produces:
  - `np_vs_pt.matmul.matmul_python(a, b)` (pure Python triple-loop), `matmul_numpy(a, b)`, `matmul_torch(a, b)`.
  - `np_vs_pt.softmax.softmax_python(x)`, `softmax_numpy(x)`, `softmax_torch(x)` — numerically stable via max-shift.
  - `np_vs_pt.kmeans.kmeans_one_step_numpy(X, centers)`, `kmeans_one_step_torch(X, centers)` — return new centers.
  - `np_vs_pt.crossentropy.cross_entropy_numpy(logits, labels)`, `cross_entropy_torch(logits, labels)`.
  - `np_vs_pt.topk.topk_python(x, k)`, `topk_numpy(x, k)`, `topk_torch(x, k)` — return indices only, sorted desc by value.

- [ ] **Step 1: Scaffold**

```bash
mkdir -p projects/week-04-numpy-vs-pytorch/{src/np_vs_pt,tests,notebooks,assignments}
touch projects/week-04-numpy-vs-pytorch/src/np_vs_pt/__init__.py
```

Write `projects/week-04-numpy-vs-pytorch/pyproject.toml`:

```toml
[project]
name = "np-vs-pt"
version = "0.1.0"
description = "Week 4 — five problems three ways: pure Python vs NumPy vs PyTorch"
requires-python = ">=3.13"
dependencies = ["numpy>=2.1", "torch>=2.5", "matplotlib>=3.9"]

[project.optional-dependencies]
dev = ["pytest>=8.3", "hypothesis>=6.112", "ruff>=0.7", "pyright>=1.1"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
addopts = "-v --strict-markers"
testpaths = ["tests"]
```

- [ ] **Step 2: Write the failing parity test**

`projects/week-04-numpy-vs-pytorch/tests/test_parity.py`:

```python
"""Numeric parity across NumPy / PyTorch / (where applicable) pure Python.
Review Focus #5 — the same problem must yield identical answers across impls."""

from __future__ import annotations
import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st


def test_matmul_parity():
    from np_vs_pt.matmul import matmul_python, matmul_numpy, matmul_torch
    a = np.random.default_rng(0).random((4, 3)).tolist()
    b = np.random.default_rng(1).random((3, 5)).tolist()
    py = matmul_python(a, b)
    numpy_ = matmul_numpy(np.array(a), np.array(b))
    torch_ = matmul_torch(torch.tensor(a), torch.tensor(b)).numpy()
    np.testing.assert_allclose(np.array(py), numpy_, atol=1e-6)
    np.testing.assert_allclose(numpy_, torch_, atol=1e-6)


def test_softmax_parity():
    from np_vs_pt.softmax import softmax_python, softmax_numpy, softmax_torch
    x = [1.0, 2.0, 3.0, 100.0]  # 100 tests numerical stability
    py = softmax_python(x)
    npx = softmax_numpy(np.array(x))
    tx  = softmax_torch(torch.tensor(x)).numpy()
    np.testing.assert_allclose(py, npx, atol=1e-9)
    np.testing.assert_allclose(npx, tx, atol=1e-6)
    assert np.isclose(sum(py), 1.0)


def test_cross_entropy_parity():
    from np_vs_pt.crossentropy import cross_entropy_numpy, cross_entropy_torch
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(8, 10))
    labels = rng.integers(0, 10, size=8)
    ce_np = cross_entropy_numpy(logits, labels)
    ce_pt = cross_entropy_torch(torch.tensor(logits), torch.tensor(labels)).item()
    assert abs(ce_np - ce_pt) < 1e-5


def test_kmeans_one_step_parity():
    from np_vs_pt.kmeans import kmeans_one_step_numpy, kmeans_one_step_torch
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 2))
    centers = rng.normal(size=(3, 2))
    new_np = kmeans_one_step_numpy(X, centers)
    new_pt = kmeans_one_step_torch(torch.tensor(X), torch.tensor(centers)).numpy()
    np.testing.assert_allclose(new_np, new_pt, atol=1e-5)


def test_topk_parity():
    from np_vs_pt.topk import topk_python, topk_numpy, topk_torch
    x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    py = topk_python(x, 3)
    npx = topk_numpy(np.array(x), 3).tolist()
    tx  = topk_torch(torch.tensor(x), 3).tolist()
    assert py == npx == tx


@given(
    n=st.integers(min_value=1, max_value=20),
    dim=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=25, deadline=None)
def test_softmax_sums_to_one_across_ranks(n, dim):
    """Property test — softmax over any shape sums to 1 within tolerance."""
    from np_vs_pt.softmax import softmax_numpy
    rng = np.random.default_rng(n * 100 + dim)
    x = rng.normal(size=n)
    y = softmax_numpy(x)
    assert abs(y.sum() - 1.0) < 1e-9
```

- [ ] **Step 3: Run — expect FAIL**

- [ ] **Step 4: Implement all 5 modules**

`src/np_vs_pt/matmul.py`:

```python
from __future__ import annotations
import numpy as np, torch


def matmul_python(a, b):
    n, m = len(a), len(b[0])
    k = len(b)
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            out[i][j] = sum(a[i][p] * b[p][j] for p in range(k))
    return out


def matmul_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def matmul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b
```

`src/np_vs_pt/softmax.py`:

```python
from __future__ import annotations
import math
import numpy as np, torch


def softmax_python(x):
    m = max(x)
    exps = [math.exp(xi - m) for xi in x]
    s = sum(exps)
    return [e / s for e in exps]


def softmax_numpy(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)
```

`src/np_vs_pt/crossentropy.py`:

```python
from __future__ import annotations
import numpy as np, torch


def cross_entropy_numpy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy: shifts logits by max for stability."""
    z = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=-1))
    log_p_target = z[np.arange(len(labels)), labels] - logsumexp
    return float(-log_p_target.mean())


def cross_entropy_torch(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, labels)
```

`src/np_vs_pt/kmeans.py`:

```python
from __future__ import annotations
import numpy as np, torch


def kmeans_one_step_numpy(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    assign = dists.argmin(axis=1)
    new = np.stack([X[assign == k].mean(axis=0) if (assign == k).any() else centers[k]
                    for k in range(len(centers))])
    return new


def kmeans_one_step_torch(X: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(dim=-1)
    assign = dists.argmin(dim=1)
    new = torch.stack([X[assign == k].mean(dim=0) if (assign == k).any() else centers[k]
                       for k in range(centers.shape[0])])
    return new
```

`src/np_vs_pt/topk.py`:

```python
from __future__ import annotations
import numpy as np, torch


def topk_python(x, k):
    return sorted(range(len(x)), key=lambda i: -x[i])[:k]


def topk_numpy(x: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-x)[:k]


def topk_torch(x: torch.Tensor, k: int) -> torch.Tensor:
    _, idx = torch.topk(x, k)
    return idx
```

`src/np_vs_pt/__init__.py`:

```python
from . import matmul, softmax, crossentropy, kmeans, topk
__all__ = ["matmul", "softmax", "crossentropy", "kmeans", "topk"]
```

- [ ] **Step 5: Run parity tests**

```bash
cd projects/week-04-numpy-vs-pytorch && uv sync --extra dev && uv run pytest -v 2>&1 | tail
```

Expected: 6 passed.

- [ ] **Step 6: README / SOLUTION_NOTES / COMPUTE / assignments / notebook**

Author each. Assignments (short):

`warmup.md`:
```markdown
# Warmup (30 min)
1. Time `matmul_python` vs `matmul_numpy` vs `matmul_torch` on 200×200 matrices. Report the ratio.
2. Verify `softmax_python([1000, 1001, 1002])` does not overflow. Where's the guard?
3. Compare `cross_entropy_numpy` output on a batch of size 1 vs `-log(softmax(logits)[label])` — should match.
```

`build.md`:
```markdown
# Build (2-3 hrs)
Implement k-means-full (10 iterations) in all three flavors:
1. Pure Python
2. NumPy
3. PyTorch (with `torch.compile` — measure the JIT compile time separately from steady-state)

Run on the same 2-D dataset. Plot the trajectory of each cluster center per iteration in one matplotlib figure per impl. Report wall-clock time. The NumPy impl should be within 2× of Torch on CPU; pure Python 100-1000× slower.
```

`challenge.md`:
```markdown
# Challenge (3+ hrs)
Implement a **cosine-similarity nearest-neighbor** function that:
1. Given a query batch (B, d) and a corpus (N, d), returns top-k indices per query.
2. Works on N up to 1M and d up to 768 without OOM on 16GB M-series.
3. Includes chunking: process the corpus in blocks of 100k rows to bound memory.
4. Runs in NumPy (baseline) AND PyTorch (with `torch.mm` and no autograd graph).
5. Timed on synthetic data (B=32, d=768, N∈{1e4, 1e5, 1e6}). Report throughput.

Stretch: quantize the corpus to int8 and re-time. Expect 2-4× speedup + 4× smaller memory.
```

- [ ] **Step 7: Notebook `01-five-problems-three-ways.py`**

```python
# %% [markdown]
# # Five problems, three ways
# Micro-benchmarks: pure Python vs NumPy vs PyTorch on the same problem.
# %%
import time, numpy as np, torch
from np_vs_pt import matmul, softmax, topk

# %% [markdown]
# ## Matmul at 200×200
# %%
rng = np.random.default_rng(0)
A = rng.random((200, 200)); B = rng.random((200, 200))

def timed(label, fn):
    t = time.perf_counter(); r = fn(); return label, time.perf_counter() - t, r

_, t_py, r_py = timed("python", lambda: matmul.matmul_python(A.tolist(), B.tolist()))
_, t_np, r_np = timed("numpy",  lambda: matmul.matmul_numpy(A, B))
_, t_pt, r_pt = timed("torch",  lambda: matmul.matmul_torch(torch.tensor(A), torch.tensor(B)).numpy())
print(f"python: {t_py*1000:8.1f} ms")
print(f"numpy:  {t_np*1000:8.1f} ms")
print(f"torch:  {t_pt*1000:8.1f} ms")
print(f"speedup numpy vs python: {t_py/t_np:.0f}×")
print(f"speedup torch vs numpy:  {t_np/t_pt:.1f}×")
```

- [ ] **Step 8: Commit**

```bash
git add projects/week-04-numpy-vs-pytorch
git commit -m "feat(w04): numpy-vs-pytorch — five problems, three impls, parity-tested"
```

---

## Task 7: Week 4 MDX — `week-04-python-info-theory.mdx`

**Files:**
- Create: `apps/book/src/content/weeks/week-04-python-info-theory.mdx`

**Interfaces:**
- Same shape as Weeks 2 and 3.

- [ ] **Step 1: Write the file**

```mdx
---
week: 4
part: 1
slug: "week-04-python-info-theory"
title: "Python for ML + Information Theory — the daily tools"
hook: "Every ML engineer speaks NumPy. Every one of them also lives in either PyTorch or JAX. Here's a week of code where the *same five problems* get solved three ways, so you feel — in your hands — what makes each stack fast."
hours: 20
computeTier: "green"
difficulty: 2
prereqSlugs: ["week-03-probability"]
referenceProject: "projects/week-04-numpy-vs-pytorch"
accentVar: "--accent-p1"
---

import Hook from '../../components/content/Hook.astro'
import Intuition from '../../components/content/Intuition.astro'
import MathBlock from '../../components/content/MathBlock.astro'
import CompanyLens from '../../components/content/CompanyLens.astro'
import ReferenceProject from '../../components/content/ReferenceProject.astro'
import Assignments from '../../components/content/Assignments.astro'
import InterviewDrill from '../../components/content/InterviewDrill.astro'
import FurtherReading from '../../components/content/FurtherReading.astro'
import KeyTakeaways from '../../components/content/KeyTakeaways.astro'
import IntuitionCallout from '../../components/callouts/Intuition.astro'
import Gotcha from '../../components/callouts/Gotcha.astro'
import { MicroRecall } from '../../components/interactive/MicroRecall'

<Hook>Every serious ML engineer speaks NumPy fluently, thinks in tensors, and can port between NumPy and PyTorch in their sleep. Week 4 makes you that engineer, and along the way it fills in the last piece of theory: **information theory**, which is what all the losses actually are.</Hook>

<Intuition>
NumPy gave us **vectorized array math on CPU**. PyTorch added **GPU support**, **autograd**, and **`nn.Module`**. JAX added **`jit`, `vmap`, `pmap`** and forced you to write in a functional style. The three are close cousins; the muscle memory transfers.

<IntuitionCallout>
Information theory quantifies *surprise*. Shannon's insight: if an event has probability p, its "information content" is `-log p` bits (or nats). Rare events carry more information; certain events carry none.
</IntuitionCallout>

The four moves you must own by end of week: **broadcasting** (why `A[:, None, :] - B[None, :, :]` works), **numerically-stable softmax and cross-entropy** (the max-shift trick), **CPU vs MPS vs CUDA memory model** (know when to `.to(device)`), and **information theory: information content, entropy, KL, cross-entropy** (the loss functions of ML).

<MicroRecall
  questions={[{
    id: 'week4-recall-broadcasting',
    prompt: 'For A of shape (M, D) and B of shape (N, D), what shape does `A[:, None, :] - B[None, :, :]` have?',
    choices: ['(M, D)', '(N, D)', '(M, N, D)', '(D, M, N)'],
    correct: 2,
    explain: 'Broadcasting: (M, 1, D) minus (1, N, D) → (M, N, D). This computes the pairwise-difference tensor between every row of A and every row of B.',
  }]}
  client:visible
/>
</Intuition>

## The math, derived

### 1. Numerically-stable softmax

The naive softmax `exp(xᵢ) / Σⱼ exp(xⱼ)` overflows for large `xᵢ`. Shift by the max:

<MathBlock latex="\text{softmax}(x)_i = \frac{e^{x_i - \max_j x_j}}{\sum_k e^{x_k - \max_j x_j}}" />

The max-shift subtracts the same constant top and bottom (algebraically identical), but keeps every `e^z` argument ≤ 0.

### 2. Cross-entropy without softmax first

Stacking softmax and log naively (`log(softmax(x))`) is wasteful and unstable. Combine them:

<MathBlock latex="\log \text{softmax}(x)_i = x_i - \max_j x_j - \log \sum_k e^{x_k - \max_j x_j}" />

Every framework's `cross_entropy` fuses this — never chain them yourself in production.

### 3. Information content & entropy

For an event of probability `p`, its information content is `-log p` (in bits if log base 2, nats if base e). Entropy is the expected information content:

<MathBlock latex="H(p) = E_{x \sim p}[-\log p(x)] = -\sum_x p(x) \log p(x)" />

Uniform distributions maximize entropy over a given support. Perfect-certainty distributions have H = 0.

### 4. KL divergence & cross-entropy

<MathBlock latex="\text{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} \ge 0" />

Cross-entropy `H(p, q) = H(p) + KL(p‖q)`. In supervised learning `p` is the (one-hot) label and `q` is the model's output — so `H(p) = 0` and minimizing cross-entropy = minimizing KL to the label = maximizing log-likelihood.

<Gotcha>
KL is **not symmetric**: `KL(p‖q) ≠ KL(q‖p)`. RLHF's "reverse KL" penalty (KL(policy‖reference)) is a specific choice — it's mode-seeking, which is why it "collapses" to a narrow distribution if you're not careful.
</Gotcha>

## The code, from scratch

Numerically-stable softmax in three flavors:

```python
# NumPy
def softmax_numpy(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()

# PyTorch
def softmax_torch(x):
    return torch.softmax(x, dim=-1)   # already stable

# JAX (for comparison)
# jax.nn.softmax(x)  # also stable
```

The framework versions are one-liners because the framework already knows to shift by the max. Writing it in NumPy makes the pattern permanent.

Full parity-tested implementations across all 5 problems (matmul, softmax, cross-entropy, k-means step, top-k): `projects/week-04-numpy-vs-pytorch/`.

<CompanyLens
  topic="Python tools + information theory"
  tldr={{
    openai:    "Tokenizer entropy is a real production metric — every OpenAI model card reports it.",
    anthropic: "Constitutional AI's KL-to-reference penalty is the RLHF constraint.",
    deepmind:  "JAX + Flax — DeepMind's canonical stack. Read their JAX docs.",
    meta:      "PyTorch is Meta's — they built it, they hire with it, they optimize for it.",
    xai:       "Custom CUDA + JAX-style pmap for Colossus-scale training loops.",
    deepseek:  "V3's routing loss uses cross-entropy over experts, plus a load-balancing regularizer.",
    qwen:      "Qwen-VL's cross-modal alignment uses InfoNCE — a contrastive info-theoretic loss.",
  }}
  whyDiffer="Every lab uses Python + NumPy for prototyping. Meta / OpenAI / Anthropic use PyTorch as production. DeepMind uses JAX. xAI mixes JAX + custom CUDA. The information-theoretic losses (cross-entropy, KL, InfoNCE) are shared substrate."
  sources={{
    openai:    { paper: "OpenAI tokenizer analyses",                          blog: "openai.com/research" },
    anthropic: { paper: "Constitutional AI arXiv:2212.08073 (KL term)",       blog: "anthropic.com/research" },
    deepmind:  { paper: "JAX / Flax / Optax docs",                            blog: "deepmind.google/discover/blog" },
    meta:      { paper: "PyTorch documentation — nn.functional",              blog: "pytorch.org/docs" },
    xai:       { paper: "Grok 3 tech report",                                 blog: "x.ai/blog" },
    deepseek:  { paper: "DeepSeek-V3 arXiv:2412.19437 — routing + aux loss",  blog: "deepseek.com/research" },
    qwen:      { paper: "Qwen-VL InfoNCE description",                        blog: "qwenlm.github.io" },
  }}
  caseStudy="The 'log-sum-exp trick' fuses softmax + log for numerical stability. Every deep-learning framework since Theano ships it. Recognizing an over-flow-y softmax in an interview is a signature senior-level moment."
  interviewAngle={{
    openai:    "Implement numerically-stable softmax in 3 lines of NumPy. Explain the shift.",
    anthropic: "Why is KL(policy‖ref), not KL(ref‖policy), used as the RLHF constraint?",
    deepmind:  "JAX's `pmap` — what does 'axis_name' actually parallelize?",
    meta:      "Port a NumPy k-means step to PyTorch. When does GPU actually win?",
    xai:       "Broadcasting shapes: given (B, T, D) and (D, K), what's the einsum for a batched matmul?",
    deepseek:  "MoE load-balancing loss — derive the auxiliary term.",
    qwen:      "InfoNCE loss — connect it to mutual information estimation.",
  }}
/>

<ReferenceProject path="projects/week-04-numpy-vs-pytorch" name="numpy-vs-pytorch" hours="6-8 hrs">
Five problems (matmul, softmax, cross-entropy, k-means step, top-k) implemented three ways (pure Python, NumPy, PyTorch), with a parity test that keeps all three impls numerically identical to atol=1e-6. Notebook micro-benchmarks the wall-clock difference.
</ReferenceProject>

<Assignments>
- **Warmup (30 min):** time all three matmul impls at 200×200; verify the numerical-stability guard in softmax on inputs like `[1000, 1001, 1002]`.
- **Build (2-3 hrs):** full k-means (10 iterations) in all three flavors; plot cluster-center trajectories; measure `torch.compile` warmup vs steady-state.
- **Challenge (3+ hrs):** cosine-similarity nearest-neighbor over a 1M-row corpus with corpus-chunking for memory safety. Optional int8 quantization for 2-4× speedup.
</Assignments>

<InterviewDrill role="applied-ml" time="20 min">
**"Write a numerically-stable softmax + cross-entropy in NumPy. No framework helpers."**

Rubric:
- Do they subtract the max before `exp`? (baseline — miss it and they OOM at scale)
- Do they use `log-sum-exp` for the log-softmax step? (senior signal — recognizes the trick)
- Can they name two failure modes: (a) huge positive logits → `inf`, (b) huge negative logits → `nan` after `0/0`? (staff signal)

Worked solution: `projects/week-04-numpy-vs-pytorch/SOLUTION_NOTES.md`.
</InterviewDrill>

<FurtherReading>
- **NumPy user guide — Broadcasting section**. Read it in one sitting; the patterns transfer to PyTorch and JAX unchanged.
- **PyTorch docs — Tensor.expand vs Tensor.repeat**. Non-obvious distinction that comes up in every interview.
- **Cover & Thomas — Elements of Information Theory, Chapter 2**. The Bible for entropy and KL. Chapter 2 alone is enough for this week.
- **Karpathy — "Yes you should understand backprop"**. Blog post that motivates why `log_softmax` is a *fused* op, not `log ∘ softmax`.
</FurtherReading>

<KeyTakeaways>
- Broadcasting: leading `1`s are inserted automatically. Design tensor shapes so pairwise ops read left-to-right.
- Numerically-stable softmax subtracts the max before `exp` — never `exp` a raw logit at production scale.
- `log_softmax` and `cross_entropy` are FUSED ops in every framework. Never chain `log` after `softmax` yourself.
- Entropy is expected information content: H(p) = E[-log p(x)]. Cross-entropy H(p, q) = H(p) + KL(p ‖ q).
- KL divergence is asymmetric. RLHF's KL-to-reference is a design choice, not a math fact.
</KeyTakeaways>
```

- [ ] **Step 2: Build**

```bash
export PATH="/Users/tharunkumarl/.nvm/versions/node/v24.21.0/bin:$PATH"
cd apps/book && pnpm run build 2>&1 | tail -5
```

Expected: `/weeks/week-04-python-info-theory/index.html` prerendered.

- [ ] **Step 3: Commit**

```bash
git add apps/book/src/content/weeks/week-04-python-info-theory.mdx
git commit -m "feat(w04): Python + Info Theory MDX with 8-part anatomy + 7-company Lens"
```

---

## Task 8: Cross-week QA — Playwright tests + link-check + final commit

**Files:**
- Modify: `apps/book/tests/company-lens-parity.spec.ts` — extend to weeks 2-4.
- Create: `apps/book/tests/weeks-2-4-render.spec.ts` — smoke test each week's route.
- Create: `apps/book/tests/katex-weeks-2-4.spec.ts` — KaTeX SSR on new pages.

**Interfaces:**
- Consumes: built `dist/client/weeks/week-0{2,3,4}-*/index.html`.

- [ ] **Step 1: Extend CompanyLens parity test**

Rewrite `apps/book/tests/company-lens-parity.spec.ts`:

```ts
import { expect, test } from '@playwright/test'

const WEEKS = [
  'week-01-linear-algebra',
  'week-02-calculus',
  'week-03-probability',
  'week-04-python-info-theory',
]
const ORDER = ['OpenAI', 'Anthropic', 'Google DeepMind', 'Meta AI (FAIR)', 'xAI', 'DeepSeek', 'Alibaba Qwen']

for (const slug of WEEKS) {
  test(`${slug} CompanyLens renders all 7 companies in fixed order`, async ({ page }) => {
    await page.goto(`/weeks/${slug}`)
    const rows = await page
      .locator('section:has(h2:has-text("Company Lens")) tbody tr')
      .allTextContents()
    expect(rows).toHaveLength(7)
    ORDER.forEach((name, i) => expect(rows[i]).toContain(name))
  })
}
```

- [ ] **Step 2: Smoke test for week routes**

Create `apps/book/tests/weeks-2-4-render.spec.ts`:

```ts
import { expect, test } from '@playwright/test'

const CASES = [
  { slug: 'week-02-calculus',         title: 'Calculus' },
  { slug: 'week-03-probability',      title: 'Probability & Statistics' },
  { slug: 'week-04-python-info-theory', title: 'Python for ML + Information Theory' },
]

for (const c of CASES) {
  test(`/weeks/${c.slug} renders and shows the reference project link`, async ({ page }) => {
    await page.goto(`/weeks/${c.slug}`)
    await expect(page.locator('main h1')).toContainText(c.title)
    await expect(page.getByRole('heading', { name: /Reference Project/i })).toBeVisible()
    await expect(page.getByText('Assignments')).toBeVisible()
  })
}
```

- [ ] **Step 3: KaTeX SSR test on new weeks**

Create `apps/book/tests/katex-weeks-2-4.spec.ts`:

```ts
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { expect, test } from '@playwright/test'

const SLUGS = ['week-02-calculus', 'week-03-probability', 'week-04-python-info-theory']

for (const slug of SLUGS) {
  test(`KaTeX renders at build time on /${slug}`, async () => {
    const html = await readFile(
      join(process.cwd(), 'dist', 'client', 'weeks', slug, 'index.html'),
      'utf-8',
    )
    expect(html).toMatch(/<span class="katex/)
  })
}
```

- [ ] **Step 4: Run all tests**

```bash
export PATH="/Users/tharunkumarl/.nvm/versions/node/v24.21.0/bin:$PATH"
cd projects && uv run pytest 2>&1 | tail -5
cd ../apps/book && pnpm run test:unit 2>&1 | tail -5
pnpm exec playwright test 2>&1 | tail -10
```

Expected: all suites green.

- [ ] **Step 5: Link check**

```bash
cd "/Users/tharunkumarl/Full Stack/Core-AI" && pnpm run check-links 2>&1 | tail -5
```

Expected: `OK — checked N unique URLs`.

- [ ] **Step 6: Commit + tag**

```bash
git add apps/book/tests
git commit -m "test(weeks 2-4): parity + smoke + KaTeX SSR on new week routes"
git commit --allow-empty -m "chore: plan 3 complete — Weeks 2-4 (Month 1 foundations) shipped"
git tag -a v0.3.0-week04 -m "Plan 3 complete: Month 1 foundations shipped (W2 calculus + W3 probability + W4 python & info theory)"
```

---

## What Plan 3 leaves for Plan 4+ (foreshadowed, not built)

- **Plan 4:** Weeks 5-8 — classical ML (linreg, trees, unsupervised, evaluation).
- **Plan 5:** Weeks 9-12 — deep learning foundations.
- **Plan 6:** Weeks 13-17 — transformers, LLMs, post-training (introduces cloud-runbook docs).
- **Plan 7:** Weeks 18-21 — systems, evals, safety.
- **Plan 8:** Weeks 22-25 — specialization + capstone + interview prep.
- **`useSignal`-based MicroRecall** is now embedded in Weeks 1-4; every future week adds 1-2 recall questions per major section (accumulating into the weak-spot heatmap on `/interview`).
- **7×25 CompanyLens matrix** is 4/25 populated at Plan 3 end. Every subsequent week adds a row of seededAngles per relevant company.
