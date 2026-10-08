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

## Strict Python typing status

`pyproject.toml` requests **strict** Pyright checking. The hardening pass completed on **2026-10-08** reduced the audit from an inherited 6,870 environment-inflated diagnostics (then 963 after the first annotation pass) to **0 diagnostics across all 27 packages**. The reduction combines:

- `py.typed` PEP-561 markers on every package.
- Narrow stub shim modules for sklearn/pandas boundaries (`_sklearn.py`, `_third_party.py`, `_typing.py`) and typed PyTorch seed/autograd adapters (`_torch.py` in each torch-using package).
- Public API signatures annotated across weeks 1-23 source files.
- Targeted file-level pragmas on educational notebooks, test fixtures, and documented optional integration boundaries (FSDP/DeepSpeed, HF generate/streamer, vLLM engine fakes, torch tensor `.tolist()`/`.backward()`/`.manual_seed`) where the stubs upstream are incomplete. Each pragma is scoped to the file whose policy carries it, not a project-wide suppression.

CI's strict Pyright step is now a **blocking gate**, not advisory, alongside the per-project runtime tests and Ruff checks. No file-wide pragma substitutes for correctness of the real signatures: pragmas suppress the specific stub-gap diagnostic categories, not reportArgumentType on load-bearing public APIs.

## Platform fixes in the final release

- Split-week and L/P/R track labels are deterministic; the 25-column matrix preserves all six specialization route links.
- WeeklyQuiz uses actual globally unique question IDs and persists review schedules; legacy progress entries remain readable.
- Reduced-motion and no-JavaScript content stays visible; formerly deferred motion browser tests now execute.
- The dev server serves the already-built Pagefind assets. Search distinguishes loading, no results and an unavailable index, rejects stale search responses, and uses a native keyboard-accessible dialog.
- Generated caches, virtual environments and harness metadata are excluded from linting; owned source remains checked.

Optional local smoke contracts validated without credentials: Week 19 CLI/demo and adapter fakes, Week 23L MCP/A2A/economics/ADK fake-module tests. Real provider/SDK calls were not made. No Vercel deployment, paid GPU job, package publication or automated outreach is part of this release.
