# Core AI — From Foundation to Frontier

A self-directed AI curriculum for full-stack engineers: mathematical foundations, classical ML, deep learning, LLMs, production systems, safety, three specialization tracks, a capstone and interview practice.

## What is included

- **29 lesson routes** covering the 25-week roadmap and every specialization choice.
- **27 Python reference packages** with tests, offline percent-format notebooks, compute guidance and three assignment levels.
- Seven company profiles and a **7 × 25 comparison matrix**, with public-source and attribution boundaries.
- Three specialization tracks: **L — LLM product/agents**, **P — applied ML**, **R — research**. All six track lessons ship; learners choose two tracks.
- Three capstone rubrics plus timed coding/debugging/take-home and outreach workbooks.
- Search, dark mode, recall questions, persistent review schedules and a browser-local interview dashboard.

W15 is split into **15a/15b**; W16 is intentionally absent. Choosing two two-week tracks takes four study weeks, and the capstone budgets about 40 hours. At 20 hours per calendar week, allow roughly **28 weeks or more**. The roadmap labels are not a promise of a 25-calendar-week finish or a guaranteed job outcome.

## Run the book

Requirements: **Node 24**, pnpm 9, **Python 3.13**, uv.

```bash
pnpm install --frozen-lockfile
uv sync --locked --all-packages --extra dev
pnpm --filter book build
pnpm --filter book dev
```

Open the local Astro URL (normally `http://localhost:4321`). Build once before using search in development: the dev server serves the generated Pagefind index from `dist/client/pagefind`. Dependencies and a cold font build may need a network connection.

## Run a reference lab

```bash
cd projects/week-21-alignment-lab
uv run --extra dev pytest
uv run python notebooks/01-alignment-lab.py
```

Each README documents its public API and exact notebook name. Required correctness tests use local fixtures, not cloud credentials or downloaded model weights. Red-tier badges apply to the optional larger-scale experiment, not a hidden paid step in the unit tests. Read `COMPUTE.md` before any optional cloud run.

## Validate the curriculum

```bash
pnpm lint
uv run --no-sync ruff check projects/ scripts/check-projects.py
uv run --no-sync python scripts/check-projects.py --notebooks
pnpm --filter @core-ai/viz test
pnpm --filter book test:unit
pnpm --filter book build
pnpm --filter book test
pnpm check-links
```

The project runner isolates pytest per package to avoid duplicate test-module collisions. Browser tests require an installed Playwright Chromium. See [validation boundaries](docs/VALIDATION.md), including the explicitly **non-passing strict Python typing audit** retained as advisory technical debt. Passing toy tests is not evidence of frontier performance, protocol certification or production safety.

## Repository map

- `apps/book/` — Astro/MDX book and browser/unit tests.
- `projects/` — 27 Python labs; each has README, source, tests, notebook, compute guide and assignments.
- `capstone/` — three complete scoring rubrics and learner deliverable contract; the learner builds the capstone.
- `interview/` — eight timed rehearsal tasks, behavioral/outreach and portfolio guidance.
- `docs/superpowers/HANDOFF.md` — release state, decisions and continuation notes.
- `docs/superpowers/specs/2026-09-22-core-ai-book-design.md` — historical design authority, clarified by the completion plan.
- `docs/superpowers/plans/2026-09-30-core-ai-completion.md` — final scope and schedule reconciliation.

No Vercel deployment, cloud GPU execution, automated outreach or package publication is performed by the curriculum. The working release branch is `plan-1-foundation`, not `main`.
