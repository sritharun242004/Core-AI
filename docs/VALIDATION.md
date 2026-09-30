# Validation boundaries

## Required release checks

Use Node 24 and Python 3.13. Install package dependencies before offline checks:

```bash
pnpm install --frozen-lockfile
uv sync --locked --all-packages --extra dev
pnpm lint
uv run --no-sync ruff check projects/ scripts/check-projects.py
uv run --no-sync python scripts/check-projects.py --notebooks
pnpm --filter @core-ai/viz test
pnpm --filter book test:unit
pnpm --filter book build
pnpm --filter book test
pnpm check-links
```

The committed `uv.lock` pins all workspace dependencies, including optional integration metadata; default dev synchronization does not enable those optional provider integrations. Each pytest process runs from its own project directory. Root collection of all projects previously collided on repeated `test_model.py`/`test_data.py` module names and used the wrong project import paths. `scripts/check-projects.py` preserves isolation and sums JUnit counts without relying on abbreviated console output.

The Python correctness suites and percent notebooks use deterministic local fixtures. Optional provider calls, real HF/vLLM inference, distributed CUDA/NCCL/FSDP/DeepSpeed execution, installed tracking services and downloaded benchmark corpora are **not** part of the tested offline path. Those integration boundaries are documented per project. Site dependency/font fetching during a cold build and the external URL checker require network access; do not describe the whole toolchain as network-free.

## Strict Python typing backlog

`pyproject.toml` still requests **strict** Pyright checking. A full audit during completion reported thousands of diagnostics (5,969 before the final track implementations), including unannotated educational NumPy/PyTorch APIs, dynamic tensor types, test doubles and absent optional-integration stubs. This is **not a passing typecheck** and is not presented as one.

CI retains the strict command as a visibly named advisory step rather than allowing this inherited, unbaselined typing backlog to obscure the mandatory per-project runtime tests and Ruff checks. The strict configuration is not weakened. Future hardening should annotate one package at a time, pin/stub optional interfaces, separate source from test typing where justified, and restore a blocking strict gate after a verified clean baseline. No runtime correctness guarantee follows from deferring these diagnostics.

## Platform fixes in the final release

- Split-week and L/P/R track labels are deterministic; the 25-column matrix preserves all six specialization route links.
- WeeklyQuiz uses actual globally unique question IDs and persists review schedules; legacy progress entries remain readable.
- Reduced-motion and no-JavaScript content stays visible; formerly deferred motion browser tests now execute.
- The dev server serves the already-built Pagefind assets. Search distinguishes loading, no results and an unavailable index, rejects stale search responses, and uses a native keyboard-accessible dialog.
- Generated caches, virtual environments and harness metadata are excluded from linting; owned source remains checked.

No Vercel deployment, paid GPU job, package publication or automated outreach is part of this release.
