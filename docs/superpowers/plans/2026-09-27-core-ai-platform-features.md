# Core AI — Plan 2: Platform-Features Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the platform-features layer of the book — the `/companies` subsystem (matrix + 7 profiles + compare), the `/interview` dashboard with client-side quiz/progress state, Motion v12 scroll-linked SVG diagrams via a new `packages/viz/` package, Pagefind client-side search, and five companion routes (`/how-to-study`, `/glossary`, `/math-primer`, `/paper-reading-protocol`, `/tech-writing`) — so every remaining week (Plans 3-8) can plug content into a finished shell without ever touching platform code.

**Architecture:** Extend the Astro 6 book with (1) a `companies` content collection lifted to first-class navigation (`/companies`, `/companies/[slug]`, `/companies/compare`) reading from seven MDX profiles with a fully-typed 5-block CompanyLens frontmatter schema; (2) a small handful of Preact + Signals islands (MicroRecall, WeeklyQuiz, InterviewDashboard, WeakSpotHeatmap, SearchDialog) that persist state to `localStorage` behind a single `lib/progress.ts` module; (3) a new workspace package `packages/viz/` exporting shared SVG primitives consumed by both MDX and future weeks; (4) Motion v12 for scroll-linked reveals wrapped in a `<ScrollReveal>` component that no-ops under `prefers-reduced-motion`; (5) Pagefind as a post-build step in Astro's build pipeline. New tests pin reduced-motion, localStorage persistence, attribution accuracy, 7-company parity, and search-index freshness — the five failure modes the spec implies but Plan 1 did not exercise.

**Tech Stack:** Astro 6 · MDX · Tailwind v4 · Preact + `@preact/signals` · Motion v12 (`motion/react`) · Pagefind v1 · Base UI (a11y primitives) · KaTeX · @astrojs/vercel · pnpm workspaces · Turborepo · Biome · Playwright · Node 24 LTS

**Spec:** `/Users/tharunkumarl/Full Stack/Core-AI/docs/superpowers/specs/2026-09-22-core-ai-book-design.md`

**Prior plan:** `/Users/tharunkumarl/Full Stack/Core-AI/docs/superpowers/plans/2026-09-22-core-ai-foundation.md` (Plan 1 — foundation + Week 1)

## Global Constraints

- **Node:** 24 LTS. **Python:** 3.13. **Framework:** Astro **6.x** (spec §7.1).
- **7 fixed companies, exact slugs & display names** (spec §4.1): `openai` OpenAI · `anthropic` Anthropic · `deepmind` Google DeepMind · `meta` Meta AI (FAIR) · `xai` xAI · `deepseek` DeepSeek · `qwen` Alibaba Qwen. Every CompanyLens block MUST render all 7 in this fixed order.
- **A11y primitives:** Base UI (successor to Radix Primitives; shadcn/ui default since July 2026 — spec §2.2). Do NOT install `@radix-ui/react-*`.
- **Motion:** Motion v12 via `motion/react` package (formerly Framer Motion — spec §2.2). All motion respects `prefers-reduced-motion: reduce` (spec §2.3).
- **Interactive islands:** Preact + `@preact/signals` only. 3-4 KB baseline budget (spec §2.2). No React, no MobX, no Zustand.
- **Persistence:** `localStorage` only. **No cross-device sync** — accepted caveat in spec §6.6. Every write goes through `lib/progress.ts`; components never touch `localStorage` directly.
- **Search:** Pagefind v1 (client-side, zero backend — spec §2.2). Runs as a `postbuild` script; no Astro adapter needed.
- **Compute-tier badge** (🟢 / 🟡 / 🔴) MUST render at the top of every week page (unchanged from Plan 1, spec §1 §3).
- **Dark mode:** first-class via `data-theme`; every component readable in both themes (spec §2.3).
- **Mobile:** nothing overflows at 375px viewport on any new route.
- **Copy discipline — VERBATIM attribution rules (spec §5.3):**
  - **DPO** — Stanford (Rafailov, Sharma, Mitchell, Ermon, Manning, Finn 2023). NOT Meta.
  - **A2A + ADK** — Google **Cloud**, NOT Google DeepMind.
  - **Attention Is All You Need** — Google Brain pre-merger (defensible under today's Google DeepMind).
  - **Chinchilla** — DeepMind (see Epoch AI replication).
  - **Colossus** — xAI cluster documented via **NVIDIA/Spectrum-X blog + HPCwire reporting**. NO peer-reviewed papers exist. Never cite a "Colossus paper."
  - **DDIA** — 2E, March 2026, Kleppmann + **Riccomini**.
  - **Age of AI** — Kissinger, Schmidt, **Huttenlocher** (3 authors).
- **No invented facts.** If a claim is not in spec §5.3 or in one of the linked primary sources, leave it blank and file a `TODO:` GitHub issue in the MDX comment. Never fill gaps by guessing.
- **Package manager:** pnpm workspaces for JS/TS; uv workspace for Python. Never mix in npm/yarn/pip lockfiles.
- **Content config path:** `apps/book/src/content.config.ts` (Astro 6 collocated pattern using `glob()` loader), NOT `src/content/config.ts`. Plan 1's earlier reference is superseded by the on-disk file.

## Review Focus

The five input classes / failure modes the spec implies but no single foundational task exercises. Each has an inline test added to the owning task.

1. **`prefers-reduced-motion`** — Motion v12 diagrams must no-op (not throw, not stutter) when the user prefers reduced motion (spec §2.3). Pin with a Playwright emulate-media test in **Task 2** that asserts `matchMedia('(prefers-reduced-motion: reduce)').matches` short-circuits the reveal.
2. **`localStorage` unavailable / private mode** — quiz progress reads/writes must survive Safari private mode and a corrupted JSON blob without crashing the island (spec §6.6). Pin with a vitest unit test in **Task 9** that stubs `localStorage.setItem` to throw + injects malformed JSON.
3. **7-company parity in CompanyLens** — every `<CompanyLens>` block must render exactly 7 rows in the fixed order (spec §4.1). Missing a company must fail loudly at build time, not silently render 6 rows. Pin with an assertion inside `CompanyLens.astro` (throws in dev + build) and a Playwright test in **Task 3**.
4. **Attribution accuracy** — the 7 verbatim attribution rules (DPO=Stanford, A2A/ADK=Google Cloud, no Colossus paper, etc.) must appear correctly and never leak the wrong attribution into any profile. Pin with a snapshot-style grep test in **Task 5** that scans all 7 company MDX files for forbidden strings ("DPO — Meta", "Colossus paper", "A2A — DeepMind", …).
5. **Pagefind index freshness** — the search index must rebuild on every `pnpm build` and include all week + company + companion routes. If a route is added and Pagefind isn't rerun, search silently returns stale results. Pin with a build-artifact test in **Task 12** that unpacks `dist/pagefind/` and grep-asserts each expected slug is present.

---

## File Structure

```
core-ai-book/                                                (repo root)
├── packages/                                                (NEW workspace)
│   └── viz/
│       ├── package.json                                     (Task 1)
│       ├── tsconfig.json                                    (Task 1)
│       ├── src/
│       │   ├── index.ts                                     (Task 1) — re-exports
│       │   ├── shared.ts                                    (Task 1) — sizing/theme helpers
│       │   ├── VectorPlayground.tsx                         (Task 1)
│       │   └── MatrixMul.tsx                                (Task 1)
│       └── tests/
│           └── viz.spec.tsx                                 (Task 1) — vitest
├── apps/book/
│   ├── package.json                                         (Task 2, 9, 12) — deps added
│   ├── astro.config.mjs                                     (Task 2, 12) — motion + pagefind
│   ├── vitest.config.ts                                     (Task 9) — for unit tests
│   ├── src/
│   │   ├── content.config.ts                                (Task 4) — extended schema
│   │   ├── content/
│   │   │   ├── companies/
│   │   │   │   ├── openai.mdx                               (Task 5) — extend existing
│   │   │   │   ├── anthropic.mdx                            (Task 5)
│   │   │   │   ├── deepmind.mdx                             (Task 5)
│   │   │   │   ├── meta.mdx                                 (Task 5)
│   │   │   │   ├── xai.mdx                                  (Task 5)
│   │   │   │   ├── deepseek.mdx                             (Task 5)
│   │   │   │   └── qwen.mdx                                 (Task 5)
│   │   │   └── extras/                                      (Task 13) — companion pages
│   │   │       ├── how-to-study.mdx
│   │   │       ├── glossary.mdx
│   │   │       ├── math-primer.mdx
│   │   │       ├── paper-reading-protocol.mdx
│   │   │       └── tech-writing.mdx
│   │   ├── lib/                                             (NEW)
│   │   │   ├── progress.ts                                  (Task 9) — localStorage store
│   │   │   ├── quiz.ts                                      (Task 9) — SM-2 spaced recall
│   │   │   └── companies.ts                                 (Task 3) — 7-company constant
│   │   ├── components/
│   │   │   ├── companies/                                   (NEW)
│   │   │   │   ├── CompanyMatrix.astro                      (Task 7)
│   │   │   │   ├── CompanyProfileHeader.astro               (Task 6)
│   │   │   │   ├── CompareTable.astro                       (Task 8)
│   │   │   │   ├── LoopGuide.astro                          (Task 6)
│   │   │   │   └── CompanyReadingList.astro                 (Task 6)
│   │   │   ├── content/
│   │   │   │   └── CompanyLens.astro                        (Task 3) — REWRITE to 5-block
│   │   │   ├── interactive/                                 (NEW)
│   │   │   │   ├── MicroRecall.tsx                          (Task 9)
│   │   │   │   ├── WeeklyQuiz.tsx                           (Task 10)
│   │   │   │   ├── InterviewDashboard.tsx                   (Task 11)
│   │   │   │   ├── WeakSpotHeatmap.tsx                      (Task 11)
│   │   │   │   ├── SearchDialog.tsx                         (Task 12)
│   │   │   │   └── ThemeToggle.tsx                          (Task 9) — used by BookShell
│   │   │   └── motion/                                      (NEW)
│   │   │       └── ScrollReveal.tsx                         (Task 2)
│   │   ├── layouts/
│   │   │   ├── CompanyLayout.astro                          (Task 6)
│   │   │   └── ExtraLayout.astro                            (Task 13)
│   │   ├── pages/
│   │   │   ├── companies/
│   │   │   │   ├── index.astro                              (Task 7)
│   │   │   │   ├── [slug].astro                             (Task 6)
│   │   │   │   └── compare.astro                            (Task 8)
│   │   │   ├── interview/
│   │   │   │   ├── index.astro                              (Task 11)
│   │   │   │   └── coding-set.astro                         (Task 11)
│   │   │   ├── how-to-study.astro                           (Task 13)
│   │   │   ├── glossary.astro                               (Task 13)
│   │   │   ├── math-primer.astro                            (Task 13)
│   │   │   ├── paper-reading-protocol.astro                 (Task 13)
│   │   │   └── tech-writing.astro                           (Task 13)
│   │   └── content/weeks/week-01-linear-algebra.mdx         (Task 14) — extend Lens to 7 rows
│   └── tests/
│       ├── companies-routes.spec.ts                         (Task 6, 7, 8)
│       ├── company-attribution.spec.ts                      (Task 5)
│       ├── interview-dashboard.spec.ts                      (Task 11)
│       ├── quiz-persistence.spec.ts                         (Task 9) — vitest
│       ├── motion-reduce.spec.ts                            (Task 2)
│       ├── search-index.spec.ts                             (Task 12)
│       └── company-lens-parity.spec.ts                      (Task 3)
├── .github/workflows/
│   ├── book.yml                                             (Task 15) — extend
│   └── link-check.yml                                       (Task 15) — extend to new routes
└── scripts/
    └── check-links.mjs                                      (Task 15) — extend URL list
```

---

## Task 1: Create the `packages/viz/` workspace package

**Files:**
- Create: `packages/viz/package.json`, `packages/viz/tsconfig.json`
- Create: `packages/viz/src/index.ts`, `packages/viz/src/shared.ts`
- Create: `packages/viz/src/VectorPlayground.tsx`, `packages/viz/src/MatrixMul.tsx`
- Create: `packages/viz/tests/viz.spec.tsx`
- Modify: `pnpm-workspace.yaml` (add `packages/*`)

**Interfaces:**
- Consumes: nothing (fresh workspace).
- Produces:
  - Package name: `@core-ai/viz` (import as `import { VectorPlayground, MatrixMul } from '@core-ai/viz'`).
  - Exports (typed): `VectorPlayground({ vectors: {x:number; y:number; label?:string; color?:string}[]; width?:number; height?:number }): JSX.Element`
  - `MatrixMul({ a: number[][]; b: number[][]; width?:number; height?:number; highlight?:{i:number;j:number} }): JSX.Element`
  - Helper: `svgColors(accent: string): { fg: string; bg: string; muted: string }` from `shared.ts`.

- [ ] **Step 1: Add `packages/*` to the workspace**

Modify `pnpm-workspace.yaml` to:

```yaml
packages:
  - "apps/*"
  - "packages/*"
```

- [ ] **Step 2: Scaffold the `packages/viz/` directory**

```bash
mkdir -p packages/viz/src packages/viz/tests
```

- [ ] **Step 3: Write `packages/viz/package.json`**

```json
{
  "name": "@core-ai/viz",
  "version": "0.1.0",
  "type": "module",
  "private": true,
  "main": "./src/index.ts",
  "types": "./src/index.ts",
  "scripts": {
    "test": "vitest run",
    "typecheck": "tsc --noEmit"
  },
  "peerDependencies": {
    "preact": "^10.24.0"
  },
  "devDependencies": {
    "preact": "^10.24.0",
    "typescript": "^5.7.0",
    "vitest": "^2.1.0",
    "@testing-library/preact": "^3.2.4",
    "jsdom": "^25.0.0"
  }
}
```

- [ ] **Step 4: Write `packages/viz/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "jsx": "preserve",
    "jsxImportSource": "preact",
    "esModuleInterop": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "resolveJsonModule": true
  },
  "include": ["src/**/*", "tests/**/*"]
}
```

- [ ] **Step 5: Write `packages/viz/src/shared.ts`**

```ts
export interface SvgColors {
  fg: string
  bg: string
  muted: string
  accent: string
}

export function svgColors(accent = 'var(--accent-p1)'): SvgColors {
  return {
    fg: 'var(--fg)',
    bg: 'var(--canvas)',
    muted: 'var(--fg-muted)',
    accent,
  }
}

export function clamp(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n))
}
```

- [ ] **Step 6: Write the failing test for `VectorPlayground`**

Create `packages/viz/tests/viz.spec.tsx`:

```tsx
/** @vitest-environment jsdom */
import { render } from '@testing-library/preact'
import { describe, expect, it } from 'vitest'
import { VectorPlayground, MatrixMul } from '../src'

describe('VectorPlayground', () => {
  it('renders one <line> per input vector', () => {
    const { container } = render(
      <VectorPlayground vectors={[{ x: 3, y: 2 }, { x: -1, y: 4 }]} />,
    )
    expect(container.querySelectorAll('line').length).toBeGreaterThanOrEqual(2)
  })

  it('labels vectors when a label is provided', () => {
    const { container } = render(
      <VectorPlayground vectors={[{ x: 1, y: 1, label: 'v₁' }]} />,
    )
    expect(container.textContent).toContain('v₁')
  })
})

describe('MatrixMul', () => {
  it('renders every cell of A and B', () => {
    const { container } = render(
      <MatrixMul a={[[1, 2], [3, 4]]} b={[[5, 6], [7, 8]]} />,
    )
    // 2×2 + 2×2 = 8 cells at minimum
    expect(container.querySelectorAll('[data-cell]').length).toBeGreaterThanOrEqual(8)
  })
})
```

- [ ] **Step 7: Run the test and confirm it fails**

```bash
pnpm --filter @core-ai/viz test
```

Expected: FAIL — `VectorPlayground is not defined`.

- [ ] **Step 8: Implement `VectorPlayground.tsx`**

```tsx
import { svgColors, clamp } from './shared'

export interface Vector {
  x: number
  y: number
  label?: string
  color?: string
}

export interface VectorPlaygroundProps {
  vectors: Vector[]
  width?: number
  height?: number
  scale?: number
}

export function VectorPlayground({
  vectors,
  width = 320,
  height = 320,
  scale = 40,
}: VectorPlaygroundProps) {
  const c = svgColors()
  const cx = width / 2
  const cy = height / 2
  const maxCoord = Math.max(1, ...vectors.flatMap((v) => [Math.abs(v.x), Math.abs(v.y)]))
  const s = clamp(scale, 10, Math.min(width, height) / (maxCoord * 2 + 1))
  return (
    <svg role="img" aria-label="Vector playground" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <line x1={0} y1={cy} x2={width} y2={cy} stroke={c.muted} strokeWidth={1} />
      <line x1={cx} y1={0} x2={cx} y2={height} stroke={c.muted} strokeWidth={1} />
      {vectors.map((v, i) => {
        const x = cx + v.x * s
        const y = cy - v.y * s
        const color = v.color ?? c.accent
        return (
          <g key={i}>
            <line x1={cx} y1={cy} x2={x} y2={y} stroke={color} strokeWidth={2} markerEnd="url(#arrow)" />
            {v.label && (
              <text x={x + 6} y={y - 6} fill={c.fg} fontSize={12} fontFamily="var(--font-mono)">
                {v.label}
              </text>
            )}
          </g>
        )
      })}
      <defs>
        <marker id="arrow" markerWidth={8} markerHeight={8} refX={7} refY={4} orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill={c.accent} />
        </marker>
      </defs>
    </svg>
  )
}
```

- [ ] **Step 9: Implement `MatrixMul.tsx`**

```tsx
import { svgColors } from './shared'

export interface MatrixMulProps {
  a: number[][]
  b: number[][]
  width?: number
  height?: number
  highlight?: { i: number; j: number }
}

export function MatrixMul({ a, b, width = 480, height = 200, highlight }: MatrixMulProps) {
  const c = svgColors()
  const cell = 32
  const gap = 24
  const rowsA = a.length
  const colsA = a[0]?.length ?? 0
  const colsB = b[0]?.length ?? 0

  const drawGrid = (m: number[][], ox: number, label: string, kind: 'a' | 'b') => (
    <g>
      <text x={ox} y={16} fill={c.muted} fontSize={12} fontFamily="var(--font-mono)">{label}</text>
      {m.map((row, i) =>
        row.map((val, j) => {
          const isHi =
            highlight &&
            ((kind === 'a' && i === highlight.i) || (kind === 'b' && j === highlight.j))
          return (
            <g key={`${kind}-${i}-${j}`} data-cell>
              <rect
                x={ox + j * cell}
                y={24 + i * cell}
                width={cell}
                height={cell}
                fill={isHi ? c.accent : 'transparent'}
                fillOpacity={isHi ? 0.15 : 0}
                stroke={c.muted}
              />
              <text
                x={ox + j * cell + cell / 2}
                y={24 + i * cell + cell / 2 + 4}
                textAnchor="middle"
                fill={c.fg}
                fontSize={12}
                fontFamily="var(--font-mono)"
              >
                {val}
              </text>
            </g>
          )
        }),
      )}
    </g>
  )

  const oxA = 8
  const oxB = oxA + colsA * cell + gap
  return (
    <svg role="img" aria-label="Matrix multiplication" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {drawGrid(a, oxA, `A (${rowsA}×${colsA})`, 'a')}
      {drawGrid(b, oxB, `B (${b.length}×${colsB})`, 'b')}
    </svg>
  )
}
```

- [ ] **Step 10: Write `packages/viz/src/index.ts`**

```ts
export { VectorPlayground, type VectorPlaygroundProps, type Vector } from './VectorPlayground'
export { MatrixMul, type MatrixMulProps } from './MatrixMul'
export { svgColors, type SvgColors } from './shared'
```

- [ ] **Step 11: Install and run the test**

```bash
pnpm install
pnpm --filter @core-ai/viz test
```

Expected: 3 passed.

- [ ] **Step 12: Wire the workspace package into the book**

Modify `apps/book/package.json` `dependencies`:

```json
"@core-ai/viz": "workspace:*"
```

Then:

```bash
pnpm install
```

- [ ] **Step 13: Commit**

```bash
git add packages pnpm-workspace.yaml apps/book/package.json pnpm-lock.yaml
git commit -m "feat(viz): scaffold @core-ai/viz with VectorPlayground + MatrixMul"
```

---

## Task 2: Motion v12 + `<ScrollReveal>` + `prefers-reduced-motion` test

**Files:**
- Create: `apps/book/src/components/motion/ScrollReveal.tsx`
- Create: `apps/book/tests/motion-reduce.spec.ts`
- Modify: `apps/book/package.json` (add `motion`)
- Modify: `apps/book/astro.config.mjs` (nothing — Motion is client-side; just ensure preact islands work)

**Interfaces:**
- Consumes: `@astrojs/preact` already installed.
- Produces: `<ScrollReveal client:visible>{...children}</ScrollReveal>` — reveals children on first scroll into view; no-ops (renders children immediately, no motion) when `prefers-reduced-motion: reduce`. Props: `{ delay?: number; y?: number; once?: boolean }`.

- [ ] **Step 1: Install Motion v12**

```bash
pnpm --filter book add motion
```

Verify: `apps/book/package.json` now has `"motion": "^12.0.0"` under dependencies.

- [ ] **Step 2: Write the failing reduced-motion test**

Create `apps/book/tests/motion-reduce.spec.ts`:

```ts
import { expect, test } from '@playwright/test'

test.describe('ScrollReveal respects prefers-reduced-motion', () => {
  test('under reduce, children are visible immediately with no transform', async ({ page, context }) => {
    await context.emulateMedia({ reducedMotion: 'reduce' })
    await page.goto('/motion-probe')
    const box = page.getByTestId('reveal-box')
    await expect(box).toBeVisible()
    const transform = await box.evaluate((el) => getComputedStyle(el).transform)
    // no motion means either 'none' or the identity matrix
    expect(['none', 'matrix(1, 0, 0, 1, 0, 0)']).toContain(transform)
    const opacity = await box.evaluate((el) => Number(getComputedStyle(el).opacity))
    expect(opacity).toBe(1)
  })

  test('without reduce, initial opacity is < 1 before viewport intersect', async ({ page, context }) => {
    await context.emulateMedia({ reducedMotion: 'no-preference' })
    await page.goto('/motion-probe?belowFold=1')
    const box = page.getByTestId('reveal-box')
    // scroll to top so the reveal box is below the viewport
    await page.evaluate(() => window.scrollTo(0, 0))
    const opacity = await box.evaluate((el) => Number(getComputedStyle(el).opacity))
    expect(opacity).toBeLessThan(1)
  })
})
```

- [ ] **Step 3: Run the test and confirm it fails**

```bash
pnpm --filter book test tests/motion-reduce.spec.ts
```

Expected: FAIL — `/motion-probe` doesn't exist.

- [ ] **Step 4: Implement `ScrollReveal.tsx`**

Create `apps/book/src/components/motion/ScrollReveal.tsx`:

```tsx
import { motion, useReducedMotion } from 'motion/react'
import type { ComponentChildren } from 'preact'

export interface ScrollRevealProps {
  children: ComponentChildren
  delay?: number
  y?: number
  once?: boolean
  'data-testid'?: string
}

export function ScrollReveal({
  children,
  delay = 0,
  y = 24,
  once = true,
  'data-testid': testId,
}: ScrollRevealProps) {
  const reduce = useReducedMotion()
  if (reduce) {
    return (
      <div data-testid={testId} style={{ opacity: 1 }}>
        {children}
      </div>
    )
  }
  return (
    <motion.div
      data-testid={testId}
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once, amount: 0.25 }}
      transition={{ duration: 0.5, delay, ease: 'easeOut' }}
    >
      {children}
    </motion.div>
  )
}
```

- [ ] **Step 5: Create the `/motion-probe` route**

Create `apps/book/src/pages/motion-probe.astro`:

```astro
---
import BookShell from '../components/layout/BookShell.astro'
import { ScrollReveal } from '../components/motion/ScrollReveal'
const url = new URL(Astro.request.url)
const belowFold = url.searchParams.has('belowFold')
---
<BookShell title="Motion probe">
  {belowFold && <div style="height: 120vh"></div>}
  <ScrollReveal client:load data-testid="reveal-box">
    <p style="padding: 24px; background: var(--canvas-subtle);">Hello, motion.</p>
  </ScrollReveal>
</BookShell>
```

*Note: `/motion-probe` is a build-time route used by tests. It's harmless in production but can be excluded from the sitemap later — for Plan 2, leaving it live is fine.*

- [ ] **Step 6: Run the test**

```bash
pnpm --filter book test tests/motion-reduce.spec.ts
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add apps/book pnpm-lock.yaml
git commit -m "feat(book): motion v12 + <ScrollReveal> with prefers-reduced-motion opt-out"
```

---

## Task 3: Rewrite `CompanyLens.astro` to the full 5-block template + 7-company parity assertion

**Files:**
- Create: `apps/book/src/lib/companies.ts` — fixed list constant
- Modify: `apps/book/src/components/content/CompanyLens.astro` — REWRITE to 5-block template
- Create: `apps/book/tests/company-lens-parity.spec.ts`

**Interfaces:**
- Consumes: nothing beyond `astro:content`.
- Produces:
  - Constant: `export const COMPANIES: readonly [{slug: CompanySlug; name: string; tint: string; emoji: string}, …7 entries…]` in fixed order.
  - Type: `export type CompanySlug = 'openai' | 'anthropic' | 'deepmind' | 'meta' | 'xai' | 'deepseek' | 'qwen'`
  - Component: `<CompanyLens topic="..." tldr={{[slug]: string}} whyDiffer={string} sources={{[slug]: {paper: string; blog: string}}} caseStudy={string} interviewAngle={{[slug]: string}}>`. Missing any of the 7 slugs in `tldr`, `sources`, or `interviewAngle` MUST throw at build time.

- [ ] **Step 1: Write `apps/book/src/lib/companies.ts`**

```ts
export type CompanySlug =
  | 'openai'
  | 'anthropic'
  | 'deepmind'
  | 'meta'
  | 'xai'
  | 'deepseek'
  | 'qwen'

export interface Company {
  slug: CompanySlug
  name: string
  tint: string
  emoji: string
}

// Fixed order per spec §4.1 — never re-sort.
export const COMPANIES: readonly Company[] = [
  { slug: 'openai',    name: 'OpenAI',           tint: '#7B61FF', emoji: '🟣' },
  { slug: 'anthropic', name: 'Anthropic',        tint: '#D97757', emoji: '🟠' },
  { slug: 'deepmind',  name: 'Google DeepMind',  tint: '#4285F4', emoji: '🔵' },
  { slug: 'meta',      name: 'Meta AI (FAIR)',   tint: '#1877F2', emoji: '🟢' },
  { slug: 'xai',       name: 'xAI',              tint: '#1E1E1E', emoji: '⚫' },
  { slug: 'deepseek',  name: 'DeepSeek',         tint: '#E4B04A', emoji: '🟡' },
  { slug: 'qwen',      name: 'Alibaba Qwen',     tint: '#FF6A00', emoji: '🟠' },
] as const

export const COMPANY_SLUGS = COMPANIES.map((c) => c.slug) as readonly CompanySlug[]

export function assertAll7<T>(record: Partial<Record<CompanySlug, T>>, block: string): asserts record is Record<CompanySlug, T> {
  const missing = COMPANY_SLUGS.filter((s) => !(s in record))
  if (missing.length > 0) {
    throw new Error(`CompanyLens ${block}: missing entries for [${missing.join(', ')}]`)
  }
}
```

- [ ] **Step 2: Rewrite `CompanyLens.astro`**

Replace `apps/book/src/components/content/CompanyLens.astro`:

```astro
---
import { COMPANIES, assertAll7, type CompanySlug } from '../../lib/companies'
interface Props {
  topic: string
  tldr: Partial<Record<CompanySlug, string>>
  whyDiffer: string
  sources: Partial<Record<CompanySlug, { paper: string; blog: string }>>
  caseStudy: string
  interviewAngle: Partial<Record<CompanySlug, string>>
}
const { topic, tldr, whyDiffer, sources, caseStudy, interviewAngle } = Astro.props
assertAll7(tldr, 'tldr')
assertAll7(sources, 'sources')
assertAll7(interviewAngle, 'interviewAngle')
---
<section class="my-10 border border-border-soft rounded-md p-6 bg-canvas-subtle">
  <h2 class="font-display text-2xl">🔍 Company Lens — {topic}</h2>

  <h3 class="font-display text-lg mt-6">TL;DR</h3>
  <div class="overflow-x-auto">
    <table class="w-full text-sm mt-2">
      <thead>
        <tr class="text-left border-b border-border-soft">
          <th class="py-2 pr-4">Lab</th>
          <th class="py-2">Their take on <em>{topic}</em></th>
        </tr>
      </thead>
      <tbody>
        {COMPANIES.map((c) => (
          <tr class="border-b border-border-soft/50">
            <td class="py-2 pr-4 whitespace-nowrap">
              <a href={`/companies/${c.slug}`} class="hover:text-accent-p1">
                <span aria-hidden="true">{c.emoji}</span> {c.name}
              </a>
            </td>
            <td class="py-2">{tldr[c.slug]}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>

  <h3 class="font-display text-lg mt-6">Why they differ</h3>
  <div class="prose-book"><p>{whyDiffer}</p></div>

  <h3 class="font-display text-lg mt-6">Primary sources</h3>
  <ul class="space-y-1 text-sm">
    {COMPANIES.map((c) => (
      <li>
        <strong>{c.name}:</strong> {sources[c.slug]?.paper} · {sources[c.slug]?.blog}
      </li>
    ))}
  </ul>

  <h3 class="font-display text-lg mt-6">Case study</h3>
  <div class="prose-book"><p>{caseStudy}</p></div>

  <h3 class="font-display text-lg mt-6">Interview angle</h3>
  <ul class="space-y-1 text-sm">
    {COMPANIES.map((c) => (
      <li><strong>{c.name}:</strong> {interviewAngle[c.slug]}</li>
    ))}
  </ul>
</section>
```

- [ ] **Step 3: Write the 7-company parity test**

Create `apps/book/tests/company-lens-parity.spec.ts`:

```ts
import { expect, test } from '@playwright/test'

test('week 1 CompanyLens renders all 7 companies in fixed order', async ({ page }) => {
  await page.goto('/weeks/week-01-linear-algebra')
  const rows = await page.locator('section:has(h2:has-text("Company Lens")) tbody tr').allTextContents()
  const order = ['OpenAI', 'Anthropic', 'Google DeepMind', 'Meta AI (FAIR)', 'xAI', 'DeepSeek', 'Alibaba Qwen']
  expect(rows).toHaveLength(7)
  order.forEach((name, i) => expect(rows[i]).toContain(name))
})
```

*Note: this test starts passing after **Task 14** wires the full 7-company Lens into `week-01-linear-algebra.mdx`. Executor: keep the test written, and expect it to fail until Task 14.*

- [ ] **Step 4: Commit**

```bash
git add apps/book/src apps/book/tests/company-lens-parity.spec.ts
git commit -m "feat(book): CompanyLens 5-block template with build-time 7-company parity check"
```

---

## Task 4: Extend `companies` collection schema for full profile

**Files:**
- Modify: `apps/book/src/content.config.ts`

**Interfaces:**
- Consumes: existing `companies` collection.
- Produces: extended Zod schema. Every profile MDX now types-checks against `{ slug, name, tint, tagline, founded, hq, philosophyLead, leaders, modelTimeline: {year: number, note: string}[], loop: {round: string, note: string}[], papers: string[], blogs: string[], books: string[], seededAngles: {week: number, note: string}[] }`.

- [ ] **Step 1: Rewrite `apps/book/src/content.config.ts`**

```ts
import { defineCollection, z } from 'astro:content'
import { glob } from 'astro/loaders'

const weeks = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/weeks' }),
  schema: z.object({
    week: z.number().int().min(1).max(25),
    part: z.number().int().min(1).max(6),
    slug: z.string(),
    title: z.string(),
    hook: z.string(),
    hours: z.number().default(20),
    computeTier: z.enum(['green', 'yellow', 'red']),
    difficulty: z.number().int().min(1).max(5),
    prereqSlugs: z.array(z.string()).default([]),
    referenceProject: z.string().optional(),
    accentVar: z.string().default('--accent-p1'),
    publishedAt: z.date().optional(),
  }),
})

const companies = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/companies' }),
  schema: z.object({
    slug: z.enum(['openai', 'anthropic', 'deepmind', 'meta', 'xai', 'deepseek', 'qwen']),
    name: z.string(),
    tint: z.string(),
    tagline: z.string(),
    founded: z.number().int(),
    hq: z.string(),
    philosophyLead: z.string(),
    leaders: z.array(z.string()).default([]),
    modelTimeline: z
      .array(z.object({ year: z.number().int(), note: z.string() }))
      .default([]),
    loop: z
      .array(z.object({ round: z.string(), note: z.string() }))
      .default([]),
    papers: z.array(z.string()).default([]),
    blogs: z.array(z.string()).default([]),
    books: z.array(z.string()).default([]),
    seededAngles: z
      .array(z.object({ week: z.number().int(), note: z.string() }))
      .default([]),
  }),
})

const extras = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/extras' }),
  schema: z.object({
    slug: z.enum(['how-to-study', 'glossary', 'math-primer', 'paper-reading-protocol', 'tech-writing']),
    title: z.string(),
    tagline: z.string(),
  }),
})

export const collections = { weeks, companies, extras }
```

- [ ] **Step 2: Verify the schema compiles**

```bash
pnpm --filter book astro sync
```

Expected: exit 0. If the existing `openai.mdx` fails because it lacks the new fields, that's acceptable — Task 5 rewrites it.

- [ ] **Step 3: Commit**

```bash
git add apps/book/src/content.config.ts
git commit -m "feat(book): extend companies schema + add extras collection"
```

---

## Task 5: Author the 7 company MDX profiles + attribution snapshot test

**Files:**
- Rewrite: `apps/book/src/content/companies/openai.mdx`
- Create: `apps/book/src/content/companies/anthropic.mdx`, `deepmind.mdx`, `meta.mdx`, `xai.mdx`, `deepseek.mdx`, `qwen.mdx`
- Create: `apps/book/tests/company-attribution.spec.ts`

**Interfaces:**
- Consumes: schema from Task 4.
- Produces: 7 MDX files whose frontmatter parses under the schema. Each body has 6 sections (Philosophy, Model timeline, Loop guide, Reading list, Books, Interview angles) mirroring the frontmatter arrays for redundancy.

- [ ] **Step 1: Write the attribution snapshot test FIRST (this is the pin)**

Create `apps/book/tests/company-attribution.spec.ts`:

```ts
import { expect, test } from '@playwright/test'
import { readFile, readdir } from 'node:fs/promises'
import { join } from 'node:path'

const FORBIDDEN_STRINGS: [string, RegExp][] = [
  ['DPO must be attributed to Stanford, not Meta',            /DPO[^.]{0,60}\bMeta\b/i],
  ['A2A/ADK must be attributed to Google Cloud, not DeepMind',/A2A[^.]{0,80}DeepMind|ADK[^.]{0,80}DeepMind/i],
  ['No Colossus peer-reviewed paper exists',                  /Colossus\s+paper|Colossus\s+arXiv/i],
  ['Chinchilla is DeepMind (not OpenAI/Anthropic)',           /Chinchilla[^.]{0,60}\b(OpenAI|Anthropic)\b/i],
  ['Age of AI has 3 authors (Kissinger, Schmidt, Huttenlocher)', /Age of AI[^.]{0,80}Kissinger[^.]{0,80}Schmidt(?!.{0,80}Huttenlocher)/i],
]

test('no company MDX file violates the attribution rules', async () => {
  const dir = join(process.cwd(), 'src/content/companies')
  const files = (await readdir(dir)).filter((f) => f.endsWith('.mdx'))
  expect(files.length).toBe(7)
  for (const f of files) {
    const body = await readFile(join(dir, f), 'utf-8')
    for (const [reason, pattern] of FORBIDDEN_STRINGS) {
      expect.soft(body).not.toMatch(pattern)
      if (pattern.test(body)) {
        throw new Error(`${f}: ${reason} — matched /${pattern.source}/`)
      }
    }
  }
})
```

- [ ] **Step 2: Run the test to confirm it initially passes (there's only 1 file and it's clean)**

```bash
pnpm --filter book test tests/company-attribution.spec.ts
```

Expected: FAIL on `expect(files.length).toBe(7)` — only `openai.mdx` exists.

- [ ] **Step 3: Rewrite `openai.mdx` under the new schema**

Replace `apps/book/src/content/companies/openai.mdx`:

```mdx
---
slug: "openai"
name: "OpenAI"
tint: "#7B61FF"
tagline: "RLHF pioneers, product-first."
founded: 2015
hq: "San Francisco, CA"
philosophyLead: "Product-first. Ship the model as the product; iterate in the open with users and feed the learnings into the next training run."
leaders:
  - "Sam Altman — CEO"
  - "Greg Brockman — President"
  - "Mira Murati — CTO (through 2024)"
modelTimeline:
  - { year: 2018, note: "GPT-1 (117M) — decoder-only transformer." }
  - { year: 2019, note: "GPT-2 (1.5B) — 'too dangerous to release.'" }
  - { year: 2020, note: "GPT-3 (175B) — in-context learning." }
  - { year: 2022, note: "InstructGPT + ChatGPT — RLHF at scale." }
  - { year: 2023, note: "GPT-4, Whisper, DALL·E 3 — multimodal push." }
  - { year: 2024, note: "o1 — first widely-deployed reasoning model." }
  - { year: 2026, note: "o3, Sora 2 — reasoning + generation converge." }
loop:
  - { round: "Recruiter screen",                note: "30 min, non-technical." }
  - { round: "Coding (ML-flavored)",            note: "No LC hard mode; think 'implement k-means from a spec'." }
  - { round: "ML debugging",                    note: "Signature-OpenAI: a training loop that misbehaves; find the bug live." }
  - { round: "ML system design",                note: "Serve LLMs at scale; caching; batching." }
  - { round: "Behavioral + values",             note: "Deployment mindset, iteration philosophy." }
papers:
  - "GPT-1/2/3/4 tech reports"
  - "InstructGPT — arXiv:2203.02155"
  - "Whisper — arXiv:2212.04356"
  - "CLIP — arXiv:2103.00020"
  - "GPT-4 System Card"
  - "o1 System Card (2024)"
blogs:
  - "openai.com/research"
books:
  - "The Age of AI — Kissinger, Schmidt, Huttenlocher"
  - "Genius Makers — Cade Metz"
seededAngles:
  - { week: 1,   note: "The dot product is what attention *is*." }
  - { week: 15,  note: "o-series and test-time compute — the design philosophy." }
  - { week: 19,  note: "Prompt caching economics." }
  - { week: 21,  note: "Values screens — how to speak about deployment." }
---

## Philosophy

Product-first. Ship the model *as* the product; iterate in the open with users and use the learnings to feed the next training run. RLHF was the first bet that paid off; the o-series and Sora extend the same "scale + human feedback + iterated deployment" arc.

## Model timeline

See frontmatter — rendered by `CompanyProfileHeader`.

## Loop guide (public information, ~2026)

5 rounds. The debugging round is signature-OpenAI: they hand you a training loop that misbehaves and ask you to find the bug live.

## Reading list

- Every GPT tech report, in order.
- **InstructGPT** — arXiv:2203.02155 — the paper that made RLHF famous.
- **Whisper** — the audio companion; a good ASR reference.
- **GPT-4 System Card** — how to read a safety card.
- Any current OpenAI flagship system card (published quarterly).

## Books

- *The Age of AI* — Kissinger, Schmidt, Huttenlocher.
- *Genius Makers* — Metz.

## Interview angles seeded in this book

- **W1** — The dot product is what attention *is*.
- **W15b** — o-series and test-time compute.
- **W19** — Prompt caching economics.
- **W21** — Values screens.
```

- [ ] **Step 4: Author `anthropic.mdx`**

Create `apps/book/src/content/companies/anthropic.mdx`:

```mdx
---
slug: "anthropic"
name: "Anthropic"
tint: "#D97757"
tagline: "Constitutional AI, MCP, interpretability, safety-first."
founded: 2021
hq: "San Francisco, CA"
philosophyLead: "Safety and interpretability as first-class engineering, not marketing. Every capability advance shipped with a system card, an RSP threshold check, and an interpretability program (Circuits) trying to understand what the model actually learned."
leaders:
  - "Dario Amodei — CEO"
  - "Daniela Amodei — President"
  - "Jared Kaplan — Chief Science Officer"
  - "Chris Olah — Interpretability lead"
modelTimeline:
  - { year: 2022, note: "Claude 1 — Constitutional AI at scale." }
  - { year: 2023, note: "Claude 2 — 100k context." }
  - { year: 2024, note: "Claude 3 (Opus / Sonnet / Haiku) — tier system." }
  - { year: 2025, note: "MCP — Model Context Protocol, open standard." }
  - { year: 2026, note: "Claude 4/5 families; RSP v3.0." }
loop:
  - { round: "Recruiter screen",             note: "Fit + interest in safety." }
  - { round: "CodeSignal take-home (90 min)",note: "Signature: Anthropic-style 90-min timed problem." }
  - { round: "ML fundamentals oral",         note: "Derivation-heavy — why does BatchNorm behave differently at train vs inference?" }
  - { round: "ML system design",             note: "Serve Claude at 100k QPS; caching; cost." }
  - { round: "Behavioral + RSP values",      note: "Where do you disagree with the RSP? Show they've read it." }
papers:
  - "Constitutional AI — arXiv:2212.08073"
  - "Toy Models of Superposition"
  - "Scaling Monosemanticity"
  - "Sleeper Agents — arXiv:2401.05566"
  - "Anthropic RSP v3.0 (Feb 2026)"
  - "MCP specification"
blogs:
  - "anthropic.com/research"
  - "transformer-circuits.pub — Chris Olah"
books:
  - "'Machines of Loving Grace' — Dario Amodei"
seededAngles:
  - { week: 15, note: "Constitutional AI vs RLHF — pick a hill, defend it." }
  - { week: 20, note: "Inspect AI + LLM-judge — Anthropic's evals stack." }
  - { week: 21, note: "SAE + logit-lens interp — how to speak Circuits fluently." }
---

## Philosophy

Safety-first interpretability. Every capability advance ships alongside a system card, an RSP threshold check, and Circuits work that tries to explain what the model actually learned.

## Model timeline

See frontmatter.

## Loop guide

5 rounds; the 90-minute CodeSignal take-home is signature-Anthropic. RSP-alignment questions in the values round — expect to name specific thresholds.

## Reading list

- Constitutional AI — arXiv:2212.08073.
- **Toy Models of Superposition** — the interpretability primer.
- **Scaling Monosemanticity** — SAE at production scale.
- **Sleeper Agents** — safety failure-mode case study.
- The MCP spec (docs.anthropic.com/mcp).

## Books

- Dario Amodei — *Machines of Loving Grace* (essay).

## Interview angles seeded in this book

- **W15** — Constitutional AI vs pure RLHF.
- **W20** — Evals stack (Inspect AI, LLM-judge).
- **W21** — Interpretability (logit lens, SAE, dictionary learning).
```

- [ ] **Step 5: Author `deepmind.mdx`**

Create `apps/book/src/content/companies/deepmind.mdx`:

```mdx
---
slug: "deepmind"
name: "Google DeepMind"
tint: "#4285F4"
tagline: "Alpha* dynasty, Gemini, TPU stack — long-horizon research."
founded: 2010
hq: "London, UK"
philosophyLead: "Foundational research with product legs. AlphaGo → AlphaFold → Gemini is one long arc: solve a hard problem in a constrained domain, generalize. Google Cloud (not DeepMind) owns A2A + ADK — a fact the interview cares that you know."
leaders:
  - "Demis Hassabis — CEO"
  - "Shane Legg — Chief AGI Scientist"
  - "Koray Kavukcuoglu — VP Research"
modelTimeline:
  - { year: 2016, note: "AlphaGo defeats Lee Sedol." }
  - { year: 2020, note: "AlphaFold 2 — protein structure prediction." }
  - { year: 2022, note: "Chinchilla — scaling law replication (DeepMind)." }
  - { year: 2023, note: "Gemini 1 — Google Brain + DeepMind merged org." }
  - { year: 2025, note: "Gemini 2.5 — long-context frontier." }
  - { year: 2026, note: "Gemini 3." }
loop:
  - { round: "Recruiter screen",         note: "Includes 'why DeepMind vs OpenAI vs Anthropic.'" }
  - { round: "Coding",                   note: "Two rounds, algorithmic + ML impl." }
  - { round: "Research discussion",      note: "Whiteboard a paper; defend design choices." }
  - { round: "ML fundamentals oral",     note: "Bayesian ML expected; JAX/Flax bonus." }
  - { round: "Behavioral + ethics",      note: "DeepMind ethics review — real, not theatre." }
papers:
  - "Attention Is All You Need — Vaswani et al., Google Brain pre-merger (2017)"
  - "AlphaGo (Silver et al.)"
  - "AlphaFold 2 (Jumper et al.)"
  - "Chinchilla (Hoffmann et al., 2022)"
  - "Gemini 1/1.5/2.5 tech reports"
  - "Griffin (recurrent linear attention)"
  - "Mixture-of-Depths"
blogs:
  - "deepmind.google/discover/blog"
  - "JAX documentation"
  - "TPU whitepapers"
books:
  - "Hassabis 2024 Nobel Prize (Chemistry) lecture — nobelprize.org"
seededAngles:
  - { week: 13, note: "Attention Is All You Need — read the original; know who wrote what." }
  - { week: 14, note: "Chinchilla scaling law — Epoch AI replication caught systematic errors." }
  - { week: 22, note: "MCP / A2A / ADK — A2A + ADK are Google Cloud, NOT DeepMind." }
---

## Philosophy

Long-horizon foundational research with product legs. AlphaGo → AlphaFold → Gemini is one arc: solve a constrained domain, generalize.

## Model timeline

See frontmatter.

## Loop guide

5 rounds. Research-discussion round is signature-DeepMind — whiteboard a paper you love, defend the design choices.

## Reading list

- **Attention Is All You Need** — Vaswani et al., 2017 (Google Brain pre-merger; today attributable under Google DeepMind).
- **AlphaFold 2** (Jumper et al.).
- **Chinchilla** — Hoffmann et al. 2022 (see the Epoch AI replication, which caught systematic errors).
- Gemini 2.5 / 3 technical reports.
- Griffin, Mixture-of-Depths.
- JAX + Flax docs; TPU whitepapers.

## Books

- Hassabis 2024 Nobel Prize (Chemistry) lecture at nobelprize.org.

## Interview angles seeded in this book

- **W13** — original transformer.
- **W14** — Chinchilla, honestly.
- **W22 (Track L)** — MCP vs A2A vs ADK — attribution matters.
```

- [ ] **Step 6: Author `meta.mdx`**

Create `apps/book/src/content/companies/meta.mdx`:

```mdx
---
slug: "meta"
name: "Meta AI (FAIR)"
tint: "#1877F2"
tagline: "Llama, PyTorch, open weights, SAM."
founded: 2013
hq: "Menlo Park, CA"
philosophyLead: "Open source as strategy. Ship weights, ship the framework (PyTorch), win by default. LeCun's position: current LLMs are not the path to AGI — see 'A Path Towards Autonomous Machine Intelligence.'"
leaders:
  - "Yann LeCun — Chief AI Scientist"
  - "Ahmad Al-Dahle — VP GenAI"
  - "Joelle Pineau — FAIR VP (through 2025)"
modelTimeline:
  - { year: 2023, note: "Llama 1 — leaked, then Llama 2 open weights." }
  - { year: 2024, note: "Llama 3 / 3.1 / 3.2 — the open-weight frontier for a moment." }
  - { year: 2025, note: "Llama 3.3." }
  - { year: 2026, note: "Llama 4 — MoE, multi-modal." }
papers:
  - "Llama 1/2/3/3.1/3.2/3.3 tech reports"
  - "DPO — Rafailov, Sharma, Mitchell, Ermon, Manning, Finn (Stanford, 2023) — Meta hires the DPO co-authors, but the paper is Stanford's"
  - "SAM / SAM-2 (Segment Anything)"
  - "ImageBind"
  - "DINOv2"
  - "Chameleon"
  - "CodeLlama"
  - "MegaBlocks"
blogs:
  - "ai.meta.com/blog"
  - "PyTorch documentation (canonical study text)"
books:
  - "LeCun — A Path Towards Autonomous Machine Intelligence (2022 position paper)"
loop:
  - { round: "Recruiter screen",       note: "Fit + Llama philosophy." }
  - { round: "DSA coding — 2 rounds",  note: "Signature-Meta: heavy pure DSA (LC medium-hard)." }
  - { round: "AI-assisted coding",     note: "Extend a repo while an interviewer watches — Cursor/Claude allowed." }
  - { round: "ML system design",       note: "Ranking at scale; feed ranking." }
  - { round: "Behavioral",             note: "Standard Meta behavioral loop." }
seededAngles:
  - { week: 15, note: "DPO — attributed to Stanford, not Meta. Meta hired the co-authors, that's the connection." }
  - { week: 18, note: "PyTorch — read the docs like a book; Meta's evaluation loop assumes fluency." }
  - { week: 22, note: "SAM/SAM-2 + ImageBind — Meta's multimodal open-weights bet." }
---

## Philosophy

Open source as strategy. Weights + framework (PyTorch) = default win. LeCun publicly disagrees with the current LLM-scaling path; JEPA + world models are FAIR's counter-bet.

## Model timeline

See frontmatter.

## Loop guide

5 rounds. Two pure-DSA rounds are unavoidable at Meta — the AI-assisted round is the new signature.

## Reading list

- Llama 3.x tech reports.
- **DPO** — Rafailov et al. 2023 — a **Stanford** paper. Common interview trap: candidates attribute it to Meta because Meta hired the co-authors.
- SAM / SAM-2.
- ImageBind.
- LeCun — *A Path Towards Autonomous Machine Intelligence* (2022).

## Books

- LeCun's position paper is required reading if you're interviewing at FAIR.

## Interview angles seeded in this book

- **W15** — DPO attribution + variants (KTO, IPO, ORPO, SimPO).
- **W18** — PyTorch depth expected.
- **W22 (Track L / P)** — SAM, ImageBind, multimodal.
```

- [ ] **Step 7: Author `xai.mdx`**

Create `apps/book/src/content/companies/xai.mdx`:

```mdx
---
slug: "xai"
name: "xAI"
tint: "#1E1E1E"
tagline: "Grok, Colossus 200k-GPU cluster, first-principles speed."
founded: 2023
hq: "Palo Alto, CA"
philosophyLead: "Compute-maximalist. Build the biggest cluster in the world (Colossus, Memphis), train fast, ship a personality-forward chatbot integrated into X. There is no Colossus paper — reporting only."
leaders:
  - "Elon Musk — CEO"
  - "Igor Babuschkin — Engineering"
  - "Tony Wu — Engineering"
modelTimeline:
  - { year: 2023, note: "Grok 1 announced." }
  - { year: 2024, note: "Grok 1.5V — vision; Colossus goes live (100k H100s)." }
  - { year: 2025, note: "Grok 2 / Grok 3; Colossus expands to 200k." }
  - { year: 2026, note: "Grok 4 / Grok-Vision — X integration." }
papers:
  - "Grok 1 / 1.5 / 2 / 3 technical documents (x.ai/blog)"
blogs:
  - "x.ai/blog"
  - "Colossus cluster — NVIDIA/Spectrum-X blog + HPCwire/DataCenterDynamics reporting (no peer-reviewed papers)"
books:
  - "Isaacson — Elon Musk (2023), chapters on xAI"
  - "Sutton — The Bitter Lesson (2019)"
loop:
  - { round: "Recruiter screen",           note: "Fast — 15 min, moves quick." }
  - { round: "Coding — 2 rounds",          note: "Live, no LC memorization." }
  - { round: "4-hour full-stack take-home",note: "Signature-xAI: build a real thing end-to-end at home." }
  - { round: "Systems design",             note: "Cluster-scale; latency; hardware-aware." }
  - { round: "Behavioral + Musk-standard bar", note: "Move-fast expected." }
seededAngles:
  - { week: 18, note: "Colossus — reporting only; there is no Colossus paper. Cite NVIDIA/Spectrum-X blog + HPCwire." }
  - { week: 19, note: "First-principles inference — Grok's serving stack." }
  - { week: 25, note: "4-hour take-home — one of the two rehearsals in prep week." }
---

## Philosophy

Compute-maximalist. Biggest cluster + first-principles training + product integration into X. LeCun disagrees loudly with the approach; xAI does not care.

## Model timeline

See frontmatter.

## Loop guide

The 4-hour take-home is signature-xAI. Coding rounds move fast — no LC memorization, "prove you can build."

## Reading list

- Grok 1 / 2 / 3 blog posts.
- **Colossus** — documented via NVIDIA/Spectrum-X blog + HPCwire / DataCenterDynamics reporting. **No peer-reviewed papers exist.** Do not cite a "Colossus paper" in an interview.
- Sutton, *The Bitter Lesson* (2019) — describes the ideology.

## Books

- Isaacson, *Elon Musk* (2023) — chapters on xAI's founding.

## Interview angles seeded in this book

- **W18** — Colossus as reporting, not paper.
- **W19** — Inference at cluster scale.
- **W25** — 4-hour take-home rehearsal.
```

- [ ] **Step 8: Author `deepseek.mdx`**

Create `apps/book/src/content/companies/deepseek.mdx`:

```mdx
---
slug: "deepseek"
name: "DeepSeek"
tint: "#E4B04A"
tagline: "V3, R1, GRPO, DualPipe — MoE + reasoning frontier."
founded: 2023
hq: "Hangzhou, China"
philosophyLead: "Efficiency at frontier scale. MoE routing tricks (V3), reasoning via RL with verifiable rewards (R1's GRPO), and infrastructure innovation (DualPipe). Papers first, weights second — reproducible research culture."
leaders:
  - "Liang Wenfeng — Founder"
modelTimeline:
  - { year: 2024, note: "DeepSeek-V2 — MoE at 236B params." }
  - { year: 2024, note: "DeepSeek-Coder." }
  - { year: 2025, note: "DeepSeek-V3 (671B MoE) + DeepSeek-R1 (GRPO, chain-of-thought RL)." }
  - { year: 2026, note: "DeepSeek-V3.5." }
papers:
  - "DeepSeek-V2 (arXiv:2405.04434)"
  - "DeepSeek-V3 (arXiv:2412.19437)"
  - "DeepSeek-R1 — arXiv:2501.12948 (introduced GRPO)"
  - "DeepSeek-Coder"
blogs:
  - "deepseek.com/research"
books: []
loop:
  - { round: "Applications closed for non-Chinese-speaking candidates in 2025; loop details are second-hand", note: "Include as awareness only." }
seededAngles:
  - { week: 15, note: "MoE routing — V3's expert allocation is the reference implementation." }
  - { week: 15, note: "GRPO — introduced in DeepSeekMath (arXiv:2402.03300) and later used by R1 (arXiv:2501.12948); read both papers before speaking about it." }
  - { week: 18, note: "DualPipe — DeepSeek's pipeline-parallel innovation." }
---

## Philosophy

Efficiency at frontier scale. MoE + reasoning + infrastructure — introduced GRPO in DeepSeekMath, later used it in R1, published DualPipe, and released weights.

## Model timeline

See frontmatter.

## Loop guide

DeepSeek's hiring loop is not publicly documented for non-Chinese-speaking candidates as of 2026. Included as a lab to study, not target.

## Reading list

- **DeepSeek-V3** — arXiv:2412.19437 — the MoE design.
- **DeepSeek-R1** — arXiv:2501.12948 — introduced **GRPO**.
- DeepSeek-Coder for code-model design choices.

## Books

None specifically about DeepSeek yet.

## Interview angles seeded in this book

- **W15b** — MoE + GRPO deep dive.
- **W18** — DualPipe.
```

- [ ] **Step 9: Author `qwen.mdx`**

Create `apps/book/src/content/companies/qwen.mdx`:

```mdx
---
slug: "qwen"
name: "Alibaba Qwen"
tint: "#FF6A00"
tagline: "Open-weight frontier from China — Qwen 3, VL, Coder."
founded: 2023
hq: "Hangzhou, China"
philosophyLead: "Open weights across sizes (0.5B → 72B → MoE), broad model family (base, instruct, code, vision, audio), and consistent publish-and-release cadence. Often grouped with Kimi K2, GLM-4.5, MiniMax-M1 as the Chinese open-frontier bucket."
leaders:
  - "Junyang Lin — Qwen team lead"
modelTimeline:
  - { year: 2023, note: "Qwen 1 — 1.8B–72B open weights." }
  - { year: 2024, note: "Qwen 2 / 2.5 — code, math, VL variants." }
  - { year: 2025, note: "Qwen 3 — MoE variant." }
  - { year: 2026, note: "Qwen 3.5 / VL 2." }
papers:
  - "Qwen technical report (arXiv:2309.16609)"
  - "Qwen 2 technical report"
  - "Qwen 2.5 technical report"
  - "Qwen-VL / Qwen-Audio reports"
blogs:
  - "qwenlm.github.io"
books: []
loop:
  - { round: "Applications closed for non-Chinese-speaking candidates in 2025-2026; publicly documented details are limited.", note: "" }
seededAngles:
  - { week: 15, note: "Open-weight family design — Qwen 3 as a 'menu' of sizes and modalities." }
  - { week: 17, note: "Qwen-VL — open multimodal reference." }
---

## Philosophy

Open weights, broad family, steady cadence. If DeepSeek is the papers-first lab, Qwen is the release-first lab of the Chinese open frontier.

## Model timeline

See frontmatter.

## Loop guide

Not publicly documented for non-Chinese-speaking candidates as of 2026.

## Reading list

- Qwen 2.5 technical report — the canonical family reference.
- Qwen-VL / Qwen-Audio for multimodal open weights.

## Books

None.

## Interview angles seeded in this book

- **W15** — Family design (menu of sizes/modalities).
- **W17** — Qwen-VL as an open multimodal reference.
```

- [ ] **Step 10: Run the attribution test**

```bash
pnpm --filter book test tests/company-attribution.spec.ts
```

Expected: PASS. Both `files.length === 7` and all 5 forbidden-string patterns return no matches.

- [ ] **Step 11: Verify the schema parses all 7**

```bash
pnpm --filter book astro sync
```

Expected: exit 0, no Zod parse errors.

- [ ] **Step 12: Commit**

```bash
git add apps/book/src/content/companies apps/book/tests/company-attribution.spec.ts
git commit -m "feat(book): 7 company profiles + attribution snapshot test"
```

---

## Task 6: `CompanyLayout.astro` + `/companies/[slug]` dynamic route

**Files:**
- Create: `apps/book/src/layouts/CompanyLayout.astro`
- Create: `apps/book/src/pages/companies/[slug].astro`
- Create: `apps/book/src/components/companies/CompanyProfileHeader.astro`, `LoopGuide.astro`, `CompanyReadingList.astro`
- Create: `apps/book/tests/companies-routes.spec.ts` (test 1: `/companies/[slug]` route)

**Interfaces:**
- Consumes: `companies` collection with the extended schema (Task 4), profile MDX (Task 5).
- Produces: 7 static routes at `/companies/openai` … `/companies/qwen`. Each renders a header (name, tagline, founded, hq, leaders), a model-timeline table, a loop-guide table, a reading list, then the MDX body slot.

- [ ] **Step 1: Write `CompanyProfileHeader.astro`**

```astro
---
interface Props {
  name: string
  tint: string
  tagline: string
  founded: number
  hq: string
  leaders: string[]
}
const { name, tint, tagline, founded, hq, leaders } = Astro.props
---
<header class="mb-8">
  <p class="text-sm text-fg-muted">Founded {founded} · {hq}</p>
  <h1 class="font-display text-5xl mt-1" style={`color: ${tint}`}>{name}</h1>
  <p class="text-md text-fg-muted italic mt-2">{tagline}</p>
  {leaders.length > 0 && (
    <p class="text-sm mt-4"><strong>Leaders:</strong> {leaders.join(' · ')}</p>
  )}
</header>
```

- [ ] **Step 2: Write `LoopGuide.astro`**

```astro
---
interface Round { round: string; note: string }
interface Props { rounds: Round[] }
const { rounds } = Astro.props
---
<section class="my-10">
  <h2 class="font-display text-2xl">Loop guide</h2>
  {rounds.length === 0 ? (
    <p class="text-fg-muted italic mt-2">Not publicly documented.</p>
  ) : (
    <ol class="mt-4 space-y-2 list-decimal list-inside">
      {rounds.map((r) => (
        <li>
          <strong>{r.round}</strong>{r.note && <> — <span class="text-fg-muted">{r.note}</span></>}
        </li>
      ))}
    </ol>
  )}
</section>
```

- [ ] **Step 3: Write `CompanyReadingList.astro`**

```astro
---
interface Props { papers: string[]; blogs: string[]; books: string[] }
const { papers, blogs, books } = Astro.props
---
<section class="my-10">
  <h2 class="font-display text-2xl">Reading list</h2>
  {papers.length > 0 && (
    <>
      <h3 class="font-display text-lg mt-4">Papers</h3>
      <ul class="mt-2 list-disc list-inside">{papers.map((p) => <li>{p}</li>)}</ul>
    </>
  )}
  {blogs.length > 0 && (
    <>
      <h3 class="font-display text-lg mt-4">Blogs</h3>
      <ul class="mt-2 list-disc list-inside">{blogs.map((b) => <li>{b}</li>)}</ul>
    </>
  )}
  {books.length > 0 && (
    <>
      <h3 class="font-display text-lg mt-4">Books</h3>
      <ul class="mt-2 list-disc list-inside">{books.map((b) => <li>{b}</li>)}</ul>
    </>
  )}
</section>
```

- [ ] **Step 4: Write `CompanyLayout.astro`**

```astro
---
import BookShell from '../components/layout/BookShell.astro'
import CompanyProfileHeader from '../components/companies/CompanyProfileHeader.astro'
import LoopGuide from '../components/companies/LoopGuide.astro'
import CompanyReadingList from '../components/companies/CompanyReadingList.astro'

interface Props {
  frontmatter: {
    slug: string
    name: string
    tint: string
    tagline: string
    founded: number
    hq: string
    leaders: string[]
    modelTimeline: { year: number; note: string }[]
    loop: { round: string; note: string }[]
    papers: string[]
    blogs: string[]
    books: string[]
    seededAngles: { week: number; note: string }[]
  }
}
const { frontmatter: f } = Astro.props
---
<BookShell title={f.name} accent={f.tint}>
  <CompanyProfileHeader name={f.name} tint={f.tint} tagline={f.tagline} founded={f.founded} hq={f.hq} leaders={f.leaders} />

  {f.modelTimeline.length > 0 && (
    <section class="my-10">
      <h2 class="font-display text-2xl">Model timeline</h2>
      <ol class="mt-4 space-y-1">
        {f.modelTimeline.map((m) => (
          <li class="flex gap-4 border-b border-border-soft py-2">
            <span class="text-fg-muted w-16 font-mono">{m.year}</span>
            <span>{m.note}</span>
          </li>
        ))}
      </ol>
    </section>
  )}

  <LoopGuide rounds={f.loop} />
  <CompanyReadingList papers={f.papers} blogs={f.blogs} books={f.books} />

  {f.seededAngles.length > 0 && (
    <section class="my-10">
      <h2 class="font-display text-2xl">Seeded interview angles</h2>
      <ul class="mt-2 space-y-1">
        {f.seededAngles.map((a) => (
          <li><strong>W{String(a.week).padStart(2, '0')}</strong> — {a.note}</li>
        ))}
      </ul>
    </section>
  )}

  <article class="prose-book mt-10"><slot /></article>
</BookShell>
```

- [ ] **Step 5: Write `pages/companies/[slug].astro`**

```astro
---
import { getCollection, type CollectionEntry } from 'astro:content'
import CompanyLayout from '../../layouts/CompanyLayout.astro'

export async function getStaticPaths() {
  const companies = await getCollection('companies')
  return companies.map((entry) => ({ params: { slug: entry.data.slug }, props: { entry } }))
}
interface Props { entry: CollectionEntry<'companies'> }
const { entry } = Astro.props
const { Content } = await entry.render()
---
<CompanyLayout frontmatter={entry.data}>
  <Content />
</CompanyLayout>
```

- [ ] **Step 6: Write `companies-routes.spec.ts`**

Create `apps/book/tests/companies-routes.spec.ts`:

```ts
import { expect, test } from '@playwright/test'

const SLUGS = ['openai', 'anthropic', 'deepmind', 'meta', 'xai', 'deepseek', 'qwen'] as const
const NAMES: Record<(typeof SLUGS)[number], string> = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  deepmind: 'Google DeepMind',
  meta: 'Meta AI (FAIR)',
  xai: 'xAI',
  deepseek: 'DeepSeek',
  qwen: 'Alibaba Qwen',
}

for (const slug of SLUGS) {
  test(`/companies/${slug} renders header + reading list`, async ({ page }) => {
    await page.goto(`/companies/${slug}`)
    await expect(page.locator('h1')).toContainText(NAMES[slug])
    await expect(page.getByRole('heading', { name: 'Reading list' })).toBeVisible()
  })
}
```

- [ ] **Step 7: Run the tests**

```bash
pnpm --filter book test tests/companies-routes.spec.ts
```

Expected: 7 passed.

- [ ] **Step 8: Commit**

```bash
git add apps/book/src apps/book/tests/companies-routes.spec.ts
git commit -m "feat(book): CompanyLayout + /companies/[slug] dynamic route"
```

---

## Task 7: `/companies` — 7×25 matrix landing

**Files:**
- Create: `apps/book/src/pages/companies/index.astro`
- Create: `apps/book/src/components/companies/CompanyMatrix.astro`
- Modify: `apps/book/tests/companies-routes.spec.ts` (add test 2)

**Interfaces:**
- Consumes: `COMPANIES` from `lib/companies.ts`, `weeks` collection.
- Produces: `/companies` route: sticky-left column of the 7 companies (with `<a>` links), sticky-top header of 25 week numbers (with `<a>` links to weeks), and a filled-in cell wherever the week's Lens seeds an angle for that company (initially just Week 1). Grid must horizontally scroll inside its own container.

- [ ] **Step 1: Write `CompanyMatrix.astro`**

```astro
---
import { COMPANIES } from '../../lib/companies'
import { getCollection } from 'astro:content'
const weeks = (await getCollection('weeks')).sort((a, b) => a.data.week - b.data.week)
const companies = await getCollection('companies')
const angles: Record<string, Record<number, string>> = Object.fromEntries(
  companies.map((c) => [c.data.slug, Object.fromEntries(c.data.seededAngles.map((a) => [a.week, a.note]))]),
)
---
<div class="overflow-x-auto border border-border-soft rounded-md">
  <table class="w-full text-sm border-separate" style="border-spacing: 0;">
    <thead>
      <tr>
        <th class="sticky left-0 top-0 z-30 bg-canvas p-2 text-left">Company \ Week</th>
        {weeks.map((w) => (
          <th class="sticky top-0 z-20 bg-canvas p-2 whitespace-nowrap">
            <a href={`/weeks/${w.data.slug}`} class="hover:text-accent-p1 text-xs">
              W{String(w.data.week).padStart(2, '0')}
            </a>
          </th>
        ))}
      </tr>
    </thead>
    <tbody>
      {COMPANIES.map((c) => (
        <tr>
          <th class="sticky left-0 z-10 bg-canvas p-2 text-left whitespace-nowrap">
            <a href={`/companies/${c.slug}`} class="hover:text-accent-p1">
              <span aria-hidden="true">{c.emoji}</span> {c.name}
            </a>
          </th>
          {weeks.map((w) => (
            <td class="p-2 border-t border-border-soft/50 align-top max-w-[240px]">
              {angles[c.slug]?.[w.data.week] ? (
                <span class="text-xs">{angles[c.slug][w.data.week]}</span>
              ) : (
                <span class="text-xs text-fg-muted/40">·</span>
              )}
            </td>
          ))}
        </tr>
      ))}
    </tbody>
  </table>
</div>
```

- [ ] **Step 2: Write `pages/companies/index.astro`**

```astro
---
import BookShell from '../../components/layout/BookShell.astro'
import CompanyMatrix from '../../components/companies/CompanyMatrix.astro'
---
<BookShell title="Companies">
  <header class="mb-8">
    <h1 class="font-display text-5xl">Companies × Weeks</h1>
    <p class="text-fg-muted mt-2">By week 25, this matrix has 7 × 25 = 175 comparison cells filled in — one interview angle per lab per week.</p>
  </header>
  <CompanyMatrix />
</BookShell>
```

- [ ] **Step 3: Extend `companies-routes.spec.ts` with the matrix test**

Append to `apps/book/tests/companies-routes.spec.ts`:

```ts
test('/companies renders 7 rows and 25 week columns', async ({ page }) => {
  await page.goto('/companies')
  const rowHeaders = page.locator('tbody th')
  await expect(rowHeaders).toHaveCount(7)
  const colHeaders = page.locator('thead th')
  // 1 corner + 25 weeks
  await expect(colHeaders).toHaveCount(26)
})

test('/companies has zero horizontal overflow on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 800 })
  await page.goto('/companies')
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(0)
})
```

- [ ] **Step 4: Run the tests**

```bash
pnpm --filter book test tests/companies-routes.spec.ts
```

Expected: 9 passed (7 profile + 2 matrix).

- [ ] **Step 5: Commit**

```bash
git add apps/book/src/pages/companies apps/book/src/components/companies apps/book/tests/companies-routes.spec.ts
git commit -m "feat(book): /companies 7×25 matrix landing"
```

---

## Task 8: `/companies/compare` — pick subset & axis

**Files:**
- Create: `apps/book/src/pages/companies/compare.astro`
- Create: `apps/book/src/components/companies/CompareTable.astro`
- Modify: `apps/book/tests/companies-routes.spec.ts` (add test 3)

**Interfaces:**
- Consumes: `COMPANIES` + `companies` collection.
- Produces: `/companies/compare?companies=openai,anthropic&axis=philosophy` renders a table with a column per selected slug and a row per selected axis. Available axes: `philosophy | timeline | loop | papers | angles`. If no `companies` param, defaults to all 7; if no `axis`, defaults to `philosophy`.

- [ ] **Step 1: Write `CompareTable.astro`**

```astro
---
import { COMPANIES, type CompanySlug } from '../../lib/companies'
import { getCollection } from 'astro:content'
interface Props { slugs: CompanySlug[]; axis: 'philosophy' | 'timeline' | 'loop' | 'papers' | 'angles' }
const { slugs, axis } = Astro.props

const all = await getCollection('companies')
const rows = slugs.map((s) => {
  const entry = all.find((c) => c.data.slug === s)
  if (!entry) throw new Error(`No profile for ${s}`)
  const d = entry.data
  const cell =
    axis === 'philosophy' ? d.philosophyLead :
    axis === 'timeline'   ? d.modelTimeline.map((m) => `${m.year}: ${m.note}`).join(' · ') :
    axis === 'loop'       ? d.loop.map((r) => r.round).join(' · ') :
    axis === 'papers'     ? d.papers.slice(0, 5).join(' · ') :
                            d.seededAngles.map((a) => `W${a.week}: ${a.note}`).join(' · ')
  return { name: d.name, tint: d.tint, cell }
})
---
<div class="overflow-x-auto border border-border-soft rounded-md my-6">
  <table class="w-full text-sm">
    <thead>
      <tr class="text-left border-b border-border-soft">
        <th class="p-3">Lab</th>
        <th class="p-3">{axis}</th>
      </tr>
    </thead>
    <tbody>
      {rows.map((r) => (
        <tr class="border-b border-border-soft/50 align-top">
          <td class="p-3 whitespace-nowrap" style={`color: ${r.tint}`}>{r.name}</td>
          <td class="p-3">{r.cell}</td>
        </tr>
      ))}
    </tbody>
  </table>
</div>
```

- [ ] **Step 2: Write `pages/companies/compare.astro`**

```astro
---
export const prerender = false // SSR: reads query params at request time
import BookShell from '../../components/layout/BookShell.astro'
import CompareTable from '../../components/companies/CompareTable.astro'
import { COMPANY_SLUGS, type CompanySlug } from '../../lib/companies'

const url = new URL(Astro.request.url)
const raw = url.searchParams.get('companies')
const requested = raw ? raw.split(',') : COMPANY_SLUGS
const slugs = requested.filter((s): s is CompanySlug => (COMPANY_SLUGS as readonly string[]).includes(s))
const axis = ((): 'philosophy' | 'timeline' | 'loop' | 'papers' | 'angles' => {
  const a = url.searchParams.get('axis')
  return a === 'timeline' || a === 'loop' || a === 'papers' || a === 'angles' ? a : 'philosophy'
})()
---
<BookShell title="Compare companies">
  <header class="mb-6">
    <h1 class="font-display text-4xl">Compare</h1>
    <p class="text-fg-muted text-sm mt-2">Pass <code>?companies=openai,anthropic</code> and <code>?axis=philosophy|timeline|loop|papers|angles</code>.</p>
  </header>
  <CompareTable slugs={slugs} axis={axis} />
  <p class="text-sm text-fg-muted mt-4">Current: <code>{slugs.join(', ')}</code> on axis <code>{axis}</code>.</p>
</BookShell>
```

*Note: `/companies/compare` is server-rendered per-request. Because Plan 1 deploys with `@astrojs/vercel`, this route builds fine as a dynamic route. Verify by checking that Astro treats it as SSR (not prerendered) in Task 16 QA.*

- [ ] **Step 3: Extend `companies-routes.spec.ts` with compare test**

Append:

```ts
test('/companies/compare defaults to all 7 companies on philosophy axis', async ({ page }) => {
  await page.goto('/companies/compare')
  const rows = page.locator('tbody tr')
  await expect(rows).toHaveCount(7)
  await expect(page.locator('code').first()).toContainText('openai')
})

test('/companies/compare filters companies via query string', async ({ page }) => {
  await page.goto('/companies/compare?companies=openai,anthropic&axis=timeline')
  const rows = page.locator('tbody tr')
  await expect(rows).toHaveCount(2)
})
```

- [ ] **Step 4: Run the tests**

```bash
pnpm --filter book test tests/companies-routes.spec.ts
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add apps/book/src apps/book/tests/companies-routes.spec.ts
git commit -m "feat(book): /companies/compare with subset + axis selection"
```

---

## Task 9: `lib/progress.ts` + `lib/quiz.ts` + `<MicroRecall>` island + persistence test

**Files:**
- Create: `apps/book/src/lib/progress.ts`, `apps/book/src/lib/quiz.ts`
- Create: `apps/book/src/components/interactive/MicroRecall.tsx`
- Create: `apps/book/src/components/interactive/ThemeToggle.tsx`
- Create: `apps/book/tests/quiz-persistence.spec.ts` (vitest)
- Create: `apps/book/vitest.config.ts`
- Modify: `apps/book/package.json` (add `vitest`, `@testing-library/preact`, `jsdom`)

**Interfaces:**
- Consumes: `@preact/signals` (already installed).
- Produces:
  - `lib/progress.ts`: `getProgress(): Progress`, `setProgress(p: Progress): void`, `recordAnswer(id: string, correct: boolean): void`, `getMastery(id: string): number` (0–1). Every function is `localStorage`-safe (`try/catch`, coerces malformed JSON to a fresh empty object).
  - `lib/quiz.ts`: `nextDueDate(current: {reps: number; ease: number; interval: number}, correct: boolean): {reps, ease, interval, dueDate: number}` — SM-2-lite spaced-recall scheduler.
  - `<MicroRecall questions={[{id, prompt, choices, correct, explain}]}>` renders one card at a time; on choice, updates progress and shows the explanation. `client:visible` island.

- [ ] **Step 1: Install vitest + jsdom**

```bash
pnpm --filter book add -D vitest @testing-library/preact jsdom
```

- [ ] **Step 2: Write `apps/book/vitest.config.ts`**

```ts
import { defineConfig } from 'vitest/config'
import preact from '@preact/preset-vite'

export default defineConfig({
  plugins: [preact()],
  test: {
    environment: 'jsdom',
    globals: false,
    include: ['tests/**/*.spec.tsx', 'tests/**/*.unit.spec.ts'],
  },
})
```

*(Also install `@preact/preset-vite`: `pnpm --filter book add -D @preact/preset-vite`.)*

- [ ] **Step 3: Add a script to `apps/book/package.json`**

```json
"scripts": {
  "dev": "astro dev",
  "build": "astro build",
  "preview": "astro preview",
  "test": "playwright test",
  "test:unit": "vitest run"
}
```

- [ ] **Step 4: Write the failing persistence unit test**

Create `apps/book/tests/quiz-persistence.unit.spec.ts`:

```ts
/** @vitest-environment jsdom */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getProgress, recordAnswer, getMastery, setProgress } from '../src/lib/progress'

describe('progress store', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
  })

  it('returns a fresh empty progress when localStorage is empty', () => {
    expect(getProgress()).toEqual({ answers: {} })
  })

  it('records answers and computes mastery in [0, 1]', () => {
    recordAnswer('q1', true)
    recordAnswer('q1', true)
    recordAnswer('q1', false)
    const m = getMastery('q1')
    expect(m).toBeGreaterThan(0)
    expect(m).toBeLessThanOrEqual(1)
  })

  it('survives malformed JSON in localStorage without throwing', () => {
    localStorage.setItem('core-ai:progress', '{{{ not json')
    expect(() => getProgress()).not.toThrow()
    expect(getProgress()).toEqual({ answers: {} })
  })

  it('survives localStorage.setItem throwing (private mode)', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('QuotaExceeded')
    })
    expect(() => setProgress({ answers: { q1: { attempts: 1, correct: 1 } } })).not.toThrow()
  })
})
```

- [ ] **Step 5: Run and confirm it fails**

```bash
pnpm --filter book test:unit
```

Expected: FAIL — module missing.

- [ ] **Step 6: Implement `lib/progress.ts`**

```ts
export interface AnswerStats { attempts: number; correct: number }
export interface Progress { answers: Record<string, AnswerStats> }

const KEY = 'core-ai:progress'

function safeParse(raw: string | null): Progress {
  if (!raw) return { answers: {} }
  try {
    const p = JSON.parse(raw)
    if (!p || typeof p !== 'object' || typeof p.answers !== 'object') return { answers: {} }
    return p as Progress
  } catch {
    return { answers: {} }
  }
}

export function getProgress(): Progress {
  if (typeof localStorage === 'undefined') return { answers: {} }
  try {
    return safeParse(localStorage.getItem(KEY))
  } catch {
    return { answers: {} }
  }
}

export function setProgress(p: Progress): void {
  if (typeof localStorage === 'undefined') return
  try {
    localStorage.setItem(KEY, JSON.stringify(p))
  } catch {
    // private mode, quota — silently drop; the in-memory copy is authoritative for this session
  }
}

export function recordAnswer(id: string, correct: boolean): void {
  const p = getProgress()
  const cur = p.answers[id] ?? { attempts: 0, correct: 0 }
  p.answers[id] = { attempts: cur.attempts + 1, correct: cur.correct + (correct ? 1 : 0) }
  setProgress(p)
}

export function getMastery(id: string): number {
  const s = getProgress().answers[id]
  if (!s || s.attempts === 0) return 0
  return s.correct / s.attempts
}
```

- [ ] **Step 7: Implement `lib/quiz.ts`**

```ts
export interface Sm2State { reps: number; ease: number; interval: number }

export interface Sm2Result extends Sm2State { dueDate: number }

// SM-2-lite: reps=# consecutive correct; ease in [1.3, 2.8]; interval in days.
export function nextDueDate(cur: Sm2State, correct: boolean, now = Date.now()): Sm2Result {
  const reps = correct ? cur.reps + 1 : 0
  const ease = Math.max(1.3, Math.min(2.8, cur.ease + (correct ? 0.1 : -0.2)))
  const interval =
    !correct   ? 1 :
    reps === 1 ? 1 :
    reps === 2 ? 3 :
                 Math.round(cur.interval * ease)
  return { reps, ease, interval, dueDate: now + interval * 86_400_000 }
}
```

- [ ] **Step 8: Run the unit test again**

```bash
pnpm --filter book test:unit
```

Expected: 4 passed.

- [ ] **Step 9: Write `MicroRecall.tsx`**

Create `apps/book/src/components/interactive/MicroRecall.tsx`:

```tsx
import { signal } from '@preact/signals'
import { recordAnswer } from '../../lib/progress'

export interface RecallQuestion {
  id: string
  prompt: string
  choices: string[]
  correct: number
  explain: string
}

export interface MicroRecallProps { questions: RecallQuestion[] }

export function MicroRecall({ questions }: MicroRecallProps) {
  const idx = signal(0)
  const pick = signal<number | null>(null)

  const q = () => questions[idx.value]
  const onPick = (i: number) => {
    pick.value = i
    recordAnswer(q().id, i === q().correct)
  }
  const next = () => {
    idx.value = (idx.value + 1) % questions.length
    pick.value = null
  }

  return (
    <aside class="my-6 rounded-md border-l-4 border-accent-p1 bg-canvas-subtle p-4">
      <div class="flex items-start gap-3">
        <span aria-hidden="true" class="text-xl">💡</span>
        <div style={{ flex: 1 }}>
          <p class="font-medium">{q().prompt}</p>
          <ul class="mt-3 space-y-2">
            {q().choices.map((c, i) => {
              const chosen = pick.value === i
              const revealed = pick.value !== null
              const isRight = i === q().correct
              const cls =
                !revealed  ? 'border-border-soft' :
                isRight    ? 'border-accent-p5 bg-accent-p5/10' :
                chosen     ? 'border-accent-p6 bg-accent-p6/10' :
                             'border-border-soft opacity-50'
              return (
                <li>
                  <button
                    type="button"
                    disabled={revealed}
                    onClick={() => onPick(i)}
                    class={`w-full text-left px-3 py-2 rounded-sm border ${cls}`}
                  >
                    {c}
                  </button>
                </li>
              )
            })}
          </ul>
          {pick.value !== null && (
            <div class="mt-3 text-sm">
              <p><strong>{pick.value === q().correct ? '✓ Correct.' : '✗ Not quite.'}</strong> {q().explain}</p>
              <button type="button" onClick={next} class="mt-2 text-accent-p1 underline">Next →</button>
            </div>
          )}
        </div>
      </div>
    </aside>
  )
}
```

- [ ] **Step 10: Write `ThemeToggle.tsx`**

Create `apps/book/src/components/interactive/ThemeToggle.tsx`:

```tsx
import { signal, effect } from '@preact/signals'

const theme = signal<'light' | 'dark'>(
  typeof document !== 'undefined' && document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light',
)

if (typeof document !== 'undefined') {
  effect(() => {
    document.documentElement.dataset.theme = theme.value
    try { localStorage.setItem('theme', theme.value) } catch {}
  })
}

export function ThemeToggle() {
  return (
    <button
      type="button"
      aria-label="Toggle theme"
      onClick={() => (theme.value = theme.value === 'dark' ? 'light' : 'dark')}
      class="px-3 py-1 rounded-sm border border-border-soft text-sm"
    >
      {theme.value === 'dark' ? '☀️' : '🌙'}
    </button>
  )
}
```

- [ ] **Step 11: Commit**

```bash
git add apps/book/src/lib apps/book/src/components/interactive apps/book/tests/quiz-persistence.unit.spec.ts apps/book/vitest.config.ts apps/book/package.json pnpm-lock.yaml
git commit -m "feat(book): lib/progress + lib/quiz + MicroRecall + ThemeToggle islands"
```

---

## Task 10: `<WeeklyQuiz>` island

**Files:**
- Create: `apps/book/src/components/interactive/WeeklyQuiz.tsx`

**Interfaces:**
- Consumes: `lib/progress.ts`, `lib/quiz.ts`.
- Produces: `<WeeklyQuiz weekId={number} questions={RecallQuestion[]} />` — 10-question quiz that shows a running score, updates SM-2 state per question, and displays a "next-due" summary on completion. Reuses `RecallQuestion` type from `MicroRecall`.

- [ ] **Step 1: Write `WeeklyQuiz.tsx`**

```tsx
import { signal, computed } from '@preact/signals'
import { recordAnswer } from '../../lib/progress'
import { nextDueDate, type Sm2State } from '../../lib/quiz'
import type { RecallQuestion } from './MicroRecall'

export interface WeeklyQuizProps { weekId: number; questions: RecallQuestion[] }

export function WeeklyQuiz({ weekId, questions }: WeeklyQuizProps) {
  const idx = signal(0)
  const answers = signal<{ i: number; correct: boolean }[]>([])
  const pick = signal<number | null>(null)

  const done = computed(() => answers.value.length === questions.length)
  const score = computed(() => answers.value.filter((a) => a.correct).length)

  const submit = (i: number) => {
    pick.value = i
    const correct = i === questions[idx.value].correct
    answers.value = [...answers.value, { i, correct }]
    recordAnswer(`week${weekId}-q${idx.value}`, correct)
  }
  const next = () => {
    idx.value = idx.value + 1
    pick.value = null
  }

  if (done.value) {
    const summary = answers.value.map((a, i) => {
      const cur: Sm2State = { reps: a.correct ? 1 : 0, ease: 2.5, interval: 1 }
      const r = nextDueDate(cur, a.correct)
      return { i, days: r.interval, correct: a.correct }
    })
    return (
      <section class="my-10 border border-border-soft rounded-md p-6">
        <h2 class="font-display text-2xl">Weekly synthesis — score {score.value} / {questions.length}</h2>
        <ul class="mt-4 space-y-1 text-sm">
          {summary.map((s) => (
            <li>
              Q{s.i + 1}: {s.correct ? '✓' : '✗'} — next due in <strong>{s.days}</strong> day{s.days === 1 ? '' : 's'}
            </li>
          ))}
        </ul>
      </section>
    )
  }

  const q = questions[idx.value]
  return (
    <section class="my-10 border border-border-soft rounded-md p-6">
      <p class="text-fg-muted text-sm">Question {idx.value + 1} of {questions.length}</p>
      <p class="font-display text-lg mt-2">{q.prompt}</p>
      <ul class="mt-3 space-y-2">
        {q.choices.map((c, i) => {
          const revealed = pick.value !== null
          const chosen = pick.value === i
          const isRight = i === q.correct
          const cls =
            !revealed  ? 'border-border-soft' :
            isRight    ? 'border-accent-p5 bg-accent-p5/10' :
            chosen     ? 'border-accent-p6 bg-accent-p6/10' :
                         'border-border-soft opacity-50'
          return (
            <li>
              <button type="button" disabled={revealed} onClick={() => submit(i)}
                class={`w-full text-left px-3 py-2 rounded-sm border ${cls}`}>
                {c}
              </button>
            </li>
          )
        })}
      </ul>
      {pick.value !== null && (
        <div class="mt-3 text-sm">
          <p>{q.explain}</p>
          <button type="button" onClick={next} class="mt-2 text-accent-p1 underline">Next →</button>
        </div>
      )}
    </section>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/book/src/components/interactive/WeeklyQuiz.tsx
git commit -m "feat(book): WeeklyQuiz island with SM-2 scheduling summary"
```

---

## Task 11: `/interview` dashboard + `WeakSpotHeatmap` + `/interview/coding-set`

**Files:**
- Create: `apps/book/src/components/interactive/InterviewDashboard.tsx`
- Create: `apps/book/src/components/interactive/WeakSpotHeatmap.tsx`
- Create: `apps/book/src/pages/interview/index.astro`
- Create: `apps/book/src/pages/interview/coding-set.astro`
- Create: `apps/book/tests/interview-dashboard.spec.ts`

**Interfaces:**
- Consumes: `lib/progress.ts`, `COMPANIES`.
- Produces:
  - `<InterviewDashboard />` — 4 tiles: total problems solved, streak (consecutive days with ≥1 answer), by-surface progress bars (Coding / SysDes / Fundamentals / Behavioral), by-company progress bars (7). All values derived from `getProgress()` — no separate stores. Client-only island.
  - `<WeakSpotHeatmap />` — a small color-graded grid of question IDs; darker = lower mastery. Clicking a cell scrolls to its owning week (`window.location.href = '/weeks/...'`).
  - `/interview` page pulls both.
  - `/interview/coding-set` page — static list of 20 tagged Leetcode-ML problems (title + difficulty + tags), grouped by tag.

- [ ] **Step 1: Extend `lib/progress.ts` with helpers**

Append to `apps/book/src/lib/progress.ts`:

```ts
// Streak/surfaces are naive derivations from `answers`. A follow-up plan can
// migrate to a richer event log; today's schema is enough for a dashboard.

export interface SurfaceCounts { coding: number; sysdes: number; fundamentals: number; behavioral: number }
export type Surface = keyof SurfaceCounts

// IDs are namespaced: `week12-q3`, `coding:linreg`, `sysdes:recsys`, `fund:batchnorm`, `beh:disagreement`.
// If the ID has no known prefix, it counts under 'fundamentals'.
export function surfaceCounts(): SurfaceCounts {
  const p = getProgress()
  const out: SurfaceCounts = { coding: 0, sysdes: 0, fundamentals: 0, behavioral: 0 }
  for (const [id, s] of Object.entries(p.answers)) {
    if (s.correct === 0) continue
    if      (id.startsWith('coding:'))    out.coding++
    else if (id.startsWith('sysdes:'))    out.sysdes++
    else if (id.startsWith('beh:'))       out.behavioral++
    else                                  out.fundamentals++
  }
  return out
}

export function totalSolved(): number {
  return Object.values(getProgress().answers).filter((a) => a.correct > 0).length
}
```

- [ ] **Step 2: Write `InterviewDashboard.tsx`**

```tsx
import { signal, effect } from '@preact/signals'
import { getProgress, surfaceCounts, totalSolved } from '../../lib/progress'
import { COMPANIES } from '../../lib/companies'

export function InterviewDashboard() {
  const tick = signal(0)
  if (typeof window !== 'undefined') {
    const onStorage = () => (tick.value = tick.value + 1)
    addEventListener('storage', onStorage)
  }

  const total = () => (tick.value, totalSolved())
  const surfaces = () => (tick.value, surfaceCounts())
  const perCompany = () => {
    tick.value
    const p = getProgress()
    const map: Record<string, number> = {}
    for (const c of COMPANIES) map[c.slug] = 0
    for (const id of Object.keys(p.answers)) {
      for (const c of COMPANIES) if (id.includes(`:${c.slug}`)) map[c.slug]++
    }
    return map
  }

  return (
    <section class="my-6 grid gap-4 md:grid-cols-2">
      <div class="rounded-md border border-border-soft p-4">
        <p class="text-sm text-fg-muted">Total problems solved</p>
        <p class="font-display text-4xl mt-1">{total()}</p>
      </div>
      <div class="rounded-md border border-border-soft p-4">
        <p class="text-sm text-fg-muted">By surface</p>
        <ul class="mt-2 text-sm space-y-1">
          {(['coding', 'sysdes', 'fundamentals', 'behavioral'] as const).map((k) => (
            <li class="flex justify-between"><span>{k}</span><span>{surfaces()[k]}</span></li>
          ))}
        </ul>
      </div>
      <div class="rounded-md border border-border-soft p-4 md:col-span-2">
        <p class="text-sm text-fg-muted">By company</p>
        <ul class="mt-2 text-sm grid grid-cols-2 md:grid-cols-4 gap-2">
          {COMPANIES.map((c) => (
            <li class="flex justify-between border-b border-border-soft/50 py-1">
              <span>{c.emoji} {c.name}</span><span>{perCompany()[c.slug]}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
```

- [ ] **Step 3: Write `WeakSpotHeatmap.tsx`**

```tsx
import { getProgress, getMastery } from '../../lib/progress'

export function WeakSpotHeatmap() {
  const ids = Object.keys(getProgress().answers).sort()
  if (ids.length === 0) {
    return <p class="text-fg-muted text-sm italic">Answer some quiz questions first to see weak spots.</p>
  }
  return (
    <section class="my-6">
      <p class="text-sm text-fg-muted mb-2">Darker = weaker (mastery below 0.5)</p>
      <div class="grid grid-cols-6 md:grid-cols-10 gap-1">
        {ids.map((id) => {
          const m = getMastery(id)
          const opacity = 0.15 + (1 - m) * 0.85
          return (
            <button
              type="button"
              key={id}
              title={`${id} — mastery ${Math.round(m * 100)}%`}
              class="aspect-square rounded-sm text-[10px] leading-none"
              style={{ background: `rgba(217, 87, 87, ${opacity})`, color: 'var(--fg)' }}
            >
              {id.slice(0, 6)}
            </button>
          )
        })}
      </div>
    </section>
  )
}
```

- [ ] **Step 4: Write `pages/interview/index.astro`**

```astro
---
import BookShell from '../../components/layout/BookShell.astro'
import { InterviewDashboard } from '../../components/interactive/InterviewDashboard'
import { WeakSpotHeatmap } from '../../components/interactive/WeakSpotHeatmap'
---
<BookShell title="Interview">
  <header class="mb-6">
    <h1 class="font-display text-5xl">Interview</h1>
    <p class="text-fg-muted mt-2 text-sm">Local-only. Progress persists in this browser's <code>localStorage</code>. No cross-device sync — accepted trade-off (spec §6.6).</p>
  </header>
  <InterviewDashboard client:only="preact" />
  <h2 class="font-display text-2xl mt-8">Weak spots</h2>
  <WeakSpotHeatmap client:only="preact" />
  <p class="mt-8"><a href="/interview/coding-set" class="text-accent-p1 underline">→ 20-problem Leetcode-ML companion set</a></p>
</BookShell>
```

- [ ] **Step 5: Write `pages/interview/coding-set.astro`**

```astro
---
import BookShell from '../../components/layout/BookShell.astro'

const set = [
  { tag: 'arrays',   items: [
    { title: 'Sliding-window max', diff: 'M' }, { title: 'Product of array except self', diff: 'M' },
    { title: 'Longest consecutive sequence', diff: 'M' }, { title: 'Trapping rain water', diff: 'H' },
  ]},
  { tag: 'strings',  items: [
    { title: 'Longest palindromic substring', diff: 'M' }, { title: 'Group anagrams', diff: 'M' },
    { title: 'Edit distance', diff: 'H' },
  ]},
  { tag: 'graphs',   items: [
    { title: 'Course schedule (topological sort)', diff: 'M' }, { title: 'Number of islands', diff: 'M' },
    { title: 'Word ladder', diff: 'H' },
  ]},
  { tag: 'ml-flavored', items: [
    { title: 'K-means one iteration', diff: 'M' }, { title: 'Softmax + cross-entropy (numerically stable)', diff: 'M' },
    { title: 'Batch matrix multiply in NumPy', diff: 'M' }, { title: 'Top-k with a heap', diff: 'M' },
    { title: 'Sample from a categorical without replacement', diff: 'M' }, { title: 'Implement a min-heap', diff: 'M' },
    { title: 'Precision @ k / Recall @ k', diff: 'E' },
  ]},
  { tag: 'systems', items: [
    { title: 'LRU cache', diff: 'M' }, { title: 'Rate limiter (token bucket)', diff: 'M' },
    { title: 'Consistent hashing', diff: 'H' },
  ]},
]
---
<BookShell title="Coding set — 20 Leetcode-ML problems">
  <h1 class="font-display text-5xl">Coding set</h1>
  <p class="text-fg-muted mt-2">20 problems tagged for the four surfaces. Solve, then log with an ID of the form <code>coding:&lt;slug&gt;</code>.</p>
  {set.map((g) => (
    <section class="my-8">
      <h2 class="font-display text-2xl">{g.tag}</h2>
      <ol class="mt-2 list-decimal list-inside space-y-1">
        {g.items.map((it) => (
          <li><strong>[{it.diff}]</strong> {it.title}</li>
        ))}
      </ol>
    </section>
  ))}
</BookShell>
```

- [ ] **Step 6: Write `interview-dashboard.spec.ts`**

```ts
import { expect, test } from '@playwright/test'

test('/interview renders the dashboard with zero progress', async ({ page }) => {
  await page.goto('/interview')
  await expect(page.getByRole('heading', { name: 'Interview' })).toBeVisible()
  await expect(page.getByText('Total problems solved')).toBeVisible()
  await expect(page.getByText('Answer some quiz questions first')).toBeVisible()
})

test('/interview reflects a seeded localStorage progress record', async ({ page }) => {
  await page.goto('/interview')
  await page.evaluate(() => {
    localStorage.setItem('core-ai:progress', JSON.stringify({
      answers: {
        'week1-q0': { attempts: 2, correct: 2 },
        'coding:lru': { attempts: 1, correct: 1 },
        'sysdes:recsys': { attempts: 1, correct: 1 },
      },
    }))
  })
  await page.reload()
  // total 3 solved
  await expect(page.locator('p.font-display.text-4xl')).toContainText('3')
})

test('/interview/coding-set has all 20 problems', async ({ page }) => {
  await page.goto('/interview/coding-set')
  const items = page.locator('ol li')
  await expect(items).toHaveCount(20)
})
```

- [ ] **Step 7: Run the tests**

```bash
pnpm --filter book test tests/interview-dashboard.spec.ts
```

Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add apps/book/src apps/book/tests/interview-dashboard.spec.ts
git commit -m "feat(book): /interview dashboard + weak-spot heatmap + coding-set page"
```

---

## Task 12: Pagefind client-side search + `<SearchDialog>` + index-freshness test

**Files:**
- Create: `apps/book/src/components/interactive/SearchDialog.tsx`
- Create: `apps/book/src/pages/search.astro`
- Create: `apps/book/tests/search-index.spec.ts`
- Modify: `apps/book/package.json` (add `pagefind` devDep + `postbuild` script)
- Modify: `apps/book/src/components/layout/BookShell.astro` (add a search-open button)

**Interfaces:**
- Consumes: post-build Pagefind index at `dist/pagefind/`.
- Produces:
  - Post-build step: `pagefind --site dist` — writes `dist/pagefind/*` after every `astro build`.
  - `<SearchDialog client:only="preact" />` — a modal `<dialog>` that dynamic-imports `/pagefind/pagefind.js` and shows top 8 results as the user types. `Cmd/Ctrl+K` opens it.
  - `/search` page — a fallback route that hosts the dialog for users who deep-link.

- [ ] **Step 1: Install Pagefind**

```bash
pnpm --filter book add -D pagefind
```

- [ ] **Step 2: Wire the postbuild script**

Modify `apps/book/package.json`:

```json
"scripts": {
  "dev": "astro dev",
  "build": "astro build",
  "postbuild": "pagefind --site dist",
  "preview": "astro preview",
  "test": "playwright test",
  "test:unit": "vitest run"
}
```

- [ ] **Step 3: Write `SearchDialog.tsx`**

```tsx
import { signal, effect } from '@preact/signals'

interface PagefindResult { id: string; data: () => Promise<{ url: string; excerpt: string; meta: { title: string } }> }
interface PagefindModule { search: (q: string) => Promise<{ results: PagefindResult[] }> }

const open = signal(false)
const query = signal('')
const hits = signal<{ url: string; title: string; excerpt: string }[]>([])
let pagefind: PagefindModule | null = null

async function ensurePagefind(): Promise<PagefindModule> {
  if (pagefind) return pagefind
  // @ts-expect-error — dynamic import of runtime asset
  pagefind = await import(/* @vite-ignore */ '/pagefind/pagefind.js')
  return pagefind!
}

async function runSearch(q: string) {
  if (!q) { hits.value = []; return }
  const pf = await ensurePagefind()
  const res = await pf.search(q)
  const first = res.results.slice(0, 8)
  const data = await Promise.all(first.map((r) => r.data()))
  hits.value = data.map((d) => ({ url: d.url, title: d.meta.title, excerpt: d.excerpt }))
}

if (typeof window !== 'undefined') {
  addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); open.value = true }
    if (e.key === 'Escape') open.value = false
  })
  effect(() => { runSearch(query.value) })
}

export function SearchDialog() {
  if (!open.value) return null
  return (
    <div role="dialog" aria-modal="true" aria-label="Search" class="fixed inset-0 z-50 bg-black/40 flex items-start justify-center p-8" onClick={() => (open.value = false)}>
      <div class="bg-canvas w-full max-w-xl rounded-md border border-border-soft p-4" onClick={(e) => e.stopPropagation()}>
        <input
          autoFocus
          type="search"
          value={query.value}
          onInput={(e) => (query.value = (e.target as HTMLInputElement).value)}
          placeholder="Search the book…"
          class="w-full bg-canvas-subtle text-fg px-3 py-2 rounded-sm border border-border-soft"
        />
        <ul class="mt-3 space-y-2">
          {hits.value.map((h) => (
            <li>
              <a href={h.url} class="block px-3 py-2 rounded-sm hover:bg-canvas-subtle">
                <p class="font-medium">{h.title}</p>
                <p class="text-sm text-fg-muted" dangerouslySetInnerHTML={{ __html: h.excerpt }} />
              </a>
            </li>
          ))}
          {query.value && hits.value.length === 0 && (
            <li class="text-fg-muted text-sm italic">No matches.</li>
          )}
        </ul>
        <p class="text-xs text-fg-muted mt-2">⌘K to open · Esc to close</p>
      </div>
    </div>
  )
}

export function SearchOpener() {
  return (
    <button type="button" onClick={() => (open.value = true)} class="text-sm px-2 py-1 rounded-sm border border-border-soft">
      ⌘K Search
    </button>
  )
}
```

- [ ] **Step 4: Write `pages/search.astro`**

```astro
---
import BookShell from '../components/layout/BookShell.astro'
import { SearchDialog } from '../components/interactive/SearchDialog'
---
<BookShell title="Search">
  <h1 class="font-display text-5xl">Search</h1>
  <p class="text-fg-muted mt-2">Press <kbd>⌘K</kbd> anywhere to search. This page hosts the same dialog for deep-linking.</p>
  <SearchDialog client:only="preact" />
  <script is:inline>
    // Auto-open on this page.
    setTimeout(() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', metaKey: true })), 200)
  </script>
</BookShell>
```

- [ ] **Step 5: Add the search opener to `BookShell.astro`**

Modify `apps/book/src/components/layout/BookShell.astro` — add the opener button next to the theme area:

Insert after `<ReadingProgressBar />`:

```astro
<div class="hidden md:flex fixed top-3 right-4 z-40 gap-2">
  <SearchOpener client:load />
  <ThemeToggle client:load />
</div>
```

Add these imports at the top:

```astro
import { SearchOpener, SearchDialog } from '../interactive/SearchDialog'
import { ThemeToggle } from '../interactive/ThemeToggle'
```

And add `<SearchDialog client:only="preact" />` before the `</body>` closing tag.

- [ ] **Step 6: Write the search-index freshness test**

Create `apps/book/tests/search-index.spec.ts`:

```ts
import { expect, test } from '@playwright/test'
import { readdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'

test('pagefind index exists and every expected route has an emitted HTML page', async () => {
  const pagefindDir = join(process.cwd(), 'dist', 'pagefind')
  const files = await readdir(pagefindDir)
  expect(files).toContain('pagefind.js')

  // Pagefind writes fragments as binary blobs; instead of parsing them,
  // confirm the underlying HTML pages Pagefind indexed actually exist in dist.
  const mustHave = [
    'index.html',
    'weeks/week-01-linear-algebra/index.html',
    'companies/index.html',
    'companies/openai/index.html', 'companies/anthropic/index.html',
    'companies/deepmind/index.html', 'companies/meta/index.html',
    'companies/xai/index.html', 'companies/deepseek/index.html',
    'companies/qwen/index.html',
    'interview/index.html', 'interview/coding-set/index.html',
    'how-to-study/index.html', 'glossary/index.html', 'math-primer/index.html',
    'paper-reading-protocol/index.html', 'tech-writing/index.html',
  ]
  for (const rel of mustHave) {
    const html = await readFile(join(process.cwd(), 'dist', rel), 'utf-8').catch(() => null)
    expect.soft(html, `expected dist/${rel} to exist for pagefind to index`).not.toBeNull()
  }
})
```

*Note: the `/how-to-study` etc. entries expect Task 13 to have shipped. Executor: this test's first pass runs after Task 13.*

- [ ] **Step 7: Commit**

```bash
git add apps/book/package.json pnpm-lock.yaml apps/book/src apps/book/tests/search-index.spec.ts
git commit -m "feat(book): pagefind search + <SearchDialog> + ⌘K keybinding"
```

---

## Task 13: Companion routes — 5 companion pages

**Files:**
- Create: `apps/book/src/content/extras/how-to-study.mdx`, `glossary.mdx`, `math-primer.mdx`, `paper-reading-protocol.mdx`, `tech-writing.mdx`
- Create: `apps/book/src/layouts/ExtraLayout.astro`
- Create: `apps/book/src/pages/how-to-study.astro`, `glossary.astro`, `math-primer.astro`, `paper-reading-protocol.astro`, `tech-writing.astro`

**Interfaces:**
- Consumes: `extras` collection (Task 4).
- Produces: 5 static routes, each rendering `ExtraLayout` around the collection's MDX body.

- [ ] **Step 1: Write `ExtraLayout.astro`**

```astro
---
import BookShell from '../components/layout/BookShell.astro'
interface Props { frontmatter: { title: string; tagline: string } }
const { frontmatter } = Astro.props
---
<BookShell title={frontmatter.title}>
  <header class="mb-8">
    <h1 class="font-display text-5xl">{frontmatter.title}</h1>
    <p class="text-fg-muted mt-2 italic">{frontmatter.tagline}</p>
  </header>
  <article class="prose-book"><slot /></article>
</BookShell>
```

- [ ] **Step 2: Write `how-to-study.mdx`**

```mdx
---
slug: "how-to-study"
title: "How to study this book"
tagline: "20-22 hours per week, 25 weeks. Here's how to spend the time."
---

## The daily loop

- **Morning (1-2 hrs):** read one section of the current week from **Intuition → Math**.
- **Afternoon (2-3 hrs):** work through **Code from scratch**, then start the **Reference Project**.
- **Evening (30-60 min):** run the **Weekly quiz** or a **MicroRecall** card. Sleep beats cramming.

## The weekly loop (20-22 hrs)

- **Mon / Tue** — sections 1-4 (Hook, Intuition, Math, Code).
- **Wed** — Company Lens + Reference Project start.
- **Thu / Fri** — Reference Project + Assignments (warmup → build → challenge).
- **Sat** — Interview Drill + Weekly Quiz.
- **Sun** — light reading + planning next week.

## The monthly loop

At the end of each Month (Parts 1-6), do the **Month Review Quiz** (spaced-recall over that month's questions).

## The whole-book loop

- **Weeks 1-4:** Math + programming foundations. **No shortcuts.**
- **Weeks 5-8:** Classical ML. Kaggle-style muscle.
- **Weeks 9-12:** Deep learning + RL primer.
- **Weeks 13-17:** Transformers + LLMs + post-training. This is where the interview questions really are.
- **Weeks 18-21:** Systems, evals, safety.
- **Weeks 22-25:** Pick 2 tracks + capstone + interview prep.

## When to skip

If a section has zero relevance to the roles you're targeting, skip it — but skim the **Hook** and **Interview Angles** anyway. Two minutes of top-of-mind is cheap.
```

- [ ] **Step 3: Write `glossary.mdx`**

```mdx
---
slug: "glossary"
title: "Glossary"
tagline: "Every term used in the book, cross-linked to its home week."
---

## A

- **Attention** — see W13.
- **ALiBi** — additive positional-bias attention scheme. W14.

## B

- **Backpropagation** — W9.
- **BatchNorm** — W9-10; typical interview trap in W10 (why does BatchNorm behave differently at inference?).

## C

- **Chinchilla scaling** — DeepMind 2022. See W14. Note the Epoch AI replication.
- **Constitutional AI** — Anthropic. W21.

## D

- **DPO** (Direct Preference Optimization) — **Stanford** 2023, Rafailov et al. W15a.
- **DualPipe** — DeepSeek pipeline-parallel innovation. W18.

## G

- **GRPO** (Group Relative Policy Optimization) — introduced in DeepSeekMath (arXiv:2402.03300), later used in DeepSeek-R1 (arXiv:2501.12948). W15b.

## L

- **LoRA** — parameter-efficient fine-tuning. W15a.

## M

- **MCP** (Model Context Protocol) — Anthropic open standard. W22.
- **MoE** (Mixture of Experts) — see W15b.

## P

- **PPO** — RL algorithm. W12, W15a.

## R

- **RLHF** — reinforcement learning from human feedback. W15a.

## S

- **SVD** — singular-value decomposition. W1.
- **SAE** (Sparse Autoencoder) — interpretability. W21.

*(This glossary is intentionally incomplete in Plan 2 — every new week's content merges its terms in.)*
```

- [ ] **Step 4: Write `math-primer.mdx`**

```mdx
---
slug: "math-primer"
title: "Math primer"
tagline: "A 4-hour refresher of the math you should already know before Week 1."
---

## What you should already know

- High-school algebra: exponents, logs, quadratic formula.
- One year of calculus: derivatives, integrals, limits.
- Basic probability: conditional probability, expected value.
- Linear algebra: what a vector is, what a matrix is, what "linearly independent" means.

## If you don't

Do these three sources in order:

1. **Khan Academy — Linear Algebra** (all playlists, ~15 hrs).
2. **3Blue1Brown — Essence of Linear Algebra** (16 videos, ~4 hrs).
3. **3Blue1Brown — Essence of Calculus** (12 videos, ~4 hrs).

Then start Week 1.

## Quick checks

- Compute $\frac{\partial}{\partial x} \sin(x^2)$ without looking it up.
- Compute $\det\begin{pmatrix}1&2\\3&4\end{pmatrix}$.
- Compute the probability of at least one 6 in 4 rolls of a fair die.

If any of these took you >30 s, do the primer first.
```

- [ ] **Step 5: Write `paper-reading-protocol.mdx`**

```mdx
---
slug: "paper-reading-protocol"
title: "Paper-reading protocol"
tagline: "How to read three papers per week without burning out."
---

## The three-pass method

### Pass 1 (10 min) — bird's eye

- Read: title, abstract, introduction, section headings, conclusion.
- Skim: figures.
- Decide: is this worth pass 2? If no, stop here.

### Pass 2 (60 min) — mechanics

- Read carefully, ignoring proofs and derivations.
- Understand the figures deeply.
- Note down: what did they do, what did they measure, what worked, what didn't.

### Pass 3 (2-4 hrs) — reproduce mentally

- Re-derive the key equations.
- Re-implement (or pseudocode) the key algorithm.
- Find one flaw or missing ablation.

## The 3-per-week rhythm

- **Monday** — 1 seminal (e.g. Attention Is All You Need).
- **Wednesday** — 1 recent (arXiv, this month).
- **Friday** — 1 orthogonal (systems / evals / safety).

## Tools

- **arxiv-sanity** — for filtering.
- **Anthropic Claude / OpenAI o-series** — as a first-pass explainer, never as ground truth.
- **Roam / Obsidian / Notion** — pick one and keep every paper as one note.
```

- [ ] **Step 6: Write `tech-writing.mdx`**

```mdx
---
slug: "tech-writing"
title: "Technical writing"
tagline: "Design docs, postmortems, blog posts — the writing that gets ML engineers hired."
---

## Three formats

### Design doc

- **Context** — what's the world today?
- **Problem** — one paragraph.
- **Non-goals** — what this doc explicitly won't do.
- **Options considered** — 2-4 approaches with tradeoffs.
- **Decision** — with rationale.
- **Rollout plan** — steps + reversal plan.

### Postmortem

- **What happened** — timeline.
- **What broke** — root cause.
- **Impact** — quantified.
- **Detection** — how we noticed.
- **Response** — what we did.
- **Follow-ups** — action items with owners.

### Blog post

- **Hook** — one striking sentence.
- **Payoff** — what the reader will know by the end.
- **Body** — 3-5 sections; every one earns its keep.
- **Code / diagram** — one high-signal artifact.
- **Takeaway** — one line the reader will remember.

## Rules

- **Kill adverbs.** Kill "really", "very", "quite", "essentially".
- **Kill hedges** except when you mean them.
- **Show don't tell** — one diagram beats three paragraphs.
- **Write it, sleep on it, rewrite it.** No first draft ships.
```

- [ ] **Step 7: Write the five page shells**

Each of `how-to-study.astro`, `glossary.astro`, `math-primer.astro`, `paper-reading-protocol.astro`, `tech-writing.astro` follows this shape (replace the slug per file):

```astro
---
import { getEntry } from 'astro:content'
import ExtraLayout from '../layouts/ExtraLayout.astro'
const entry = await getEntry('extras', 'how-to-study')
if (!entry) throw new Error('missing extras entry: how-to-study')
const { Content } = await entry.render()
---
<ExtraLayout frontmatter={entry.data}>
  <Content />
</ExtraLayout>
```

Write all 5, replacing `'how-to-study'` with the matching slug.

- [ ] **Step 8: Verify the routes**

```bash
pnpm --filter book astro sync
pnpm --filter book dev &
sleep 4
for s in how-to-study glossary math-primer paper-reading-protocol tech-writing; do
  curl -s -o /dev/null -w "%{http_code} /$s\n" http://localhost:4321/$s
done
kill %1
```

Expected: 5 lines, all `200`.

- [ ] **Step 9: Run the search-index test now that all routes are seeded**

```bash
pnpm --filter book build
pnpm --filter book test tests/search-index.spec.ts
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add apps/book/src
git commit -m "feat(book): 5 companion routes + extras collection"
```

---

## Task 14: Wire the full 7-company CompanyLens into Week 1

**Files:**
- Modify: `apps/book/src/content/weeks/week-01-linear-algebra.mdx`

**Interfaces:**
- Consumes: `<CompanyLens>` (Task 3) — now requires all 7 slugs in `tldr`, `sources`, `interviewAngle`.
- Produces: Week 1's Lens block satisfies the parity assertion, so `tests/company-lens-parity.spec.ts` from Task 3 passes.

- [ ] **Step 1: Read the existing Week 1 Lens block**

```bash
grep -n "CompanyLens" apps/book/src/content/weeks/week-01-linear-algebra.mdx | head
```

Locate the current `<CompanyLens ... />` invocation and its surrounding block.

- [ ] **Step 2: Replace the Lens block with the full 5-block, 7-company version**

Replace the existing Lens block with:

```mdx
<CompanyLens
  topic="Linear algebra"
  tldr={{
    openai:    "Dot product is what attention is. Everything scales from there.",
    anthropic: "Same substrate; safety layer sits on top of the same tensor math.",
    deepmind:  "Matrix decompositions (SVD, Cholesky) show up everywhere in JAX code.",
    meta:      "PyTorch is a linear-algebra library first; everything else is layered.",
    xai:       "Cluster-scale matmuls; the linear algebra is unchanged, the hardware isn't.",
    deepseek:  "MoE routing is a sparse linear projection; understand it before V3.",
    qwen:      "Open-weight matrices at every family size — you can literally inspect them.",
  }}
  whyDiffer="All labs agree that linear algebra is the substrate; they differ in what they emphasize downstream — RLHF (OpenAI, Anthropic), scaling laws (DeepMind), tooling (Meta/PyTorch), cluster efficiency (xAI), routing (DeepSeek), and open-weight breadth (Qwen)."
  sources={{
    openai:    { paper: "GPT-1 tech report",              blog: "openai.com/research" },
    anthropic: { paper: "Toy Models of Superposition",    blog: "transformer-circuits.pub" },
    deepmind:  { paper: "Attention Is All You Need (Vaswani et al., 2017, Google Brain pre-merger)", blog: "deepmind.google/discover/blog" },
    meta:      { paper: "Llama 3 tech report",            blog: "ai.meta.com/blog" },
    xai:       { paper: "Grok 3 technical report",        blog: "x.ai/blog" },
    deepseek:  { paper: "DeepSeek-V3 (arXiv:2412.19437)", blog: "deepseek.com/research" },
    qwen:      { paper: "Qwen 2.5 technical report",      blog: "qwenlm.github.io" },
  }}
  caseStudy="SVD image compression (this week's reference project) is a working demonstration of low-rank approximation, the same trick LoRA uses in W15 to make a 7B model fine-tunable on a single GPU."
  interviewAngle={{
    openai:    "Derive attention from the dot product.",
    anthropic: "How does a linear-probing SAE work — why the linear part?",
    deepmind:  "Chinchilla-style scaling law fits; what's the linear model?",
    meta:      "Implement matmul in NumPy without loops; verify against @.",
    xai:       "Cluster comms: why does ring-all-reduce beat all-to-all?",
    deepseek:  "Sparse expert routing — write the top-k gate.",
    qwen:      "Given a family of sizes, how would you compare per-parameter compute?",
  }}
/>
```

- [ ] **Step 3: Run the parity test**

```bash
pnpm --filter book test tests/company-lens-parity.spec.ts
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add apps/book/src/content/weeks/week-01-linear-algebra.mdx
git commit -m "feat(w01): full 7-company CompanyLens block"
```

---

## Task 15: Extend CI — routes + link check + a11y regression

**Files:**
- Modify: `.github/workflows/book.yml` — add `pnpm test:unit`, add Pagefind build step
- Modify: `.github/workflows/link-check.yml` (or `scripts/check-links.mjs`) — add new routes

**Interfaces:**
- Consumes: everything.
- Produces: CI green on the new routes + tests.

- [ ] **Step 1: Read the current book workflow**

```bash
cat .github/workflows/book.yml
```

Identify where `pnpm --filter book build` and `pnpm --filter book test` run.

- [ ] **Step 2: Add a `test:unit` step**

Modify `.github/workflows/book.yml` so that between "Install dependencies" and "Build book", the pipeline now runs:

```yaml
      - name: Vitest — unit tests
        working-directory: apps/book
        run: pnpm run test:unit

      - name: Build book (includes pagefind postbuild)
        working-directory: apps/book
        run: pnpm run build

      - name: Playwright — integration tests
        working-directory: apps/book
        run: pnpm run test
```

*(Adjust names to match whatever's in the file — the intent is: unit → build → integration.)*

- [ ] **Step 3: Extend `scripts/check-links.mjs`**

Open `scripts/check-links.mjs`. Locate the list of URLs it walks. Add the following routes to the walk (relative to the site origin):

```
/companies
/companies/openai
/companies/anthropic
/companies/deepmind
/companies/meta
/companies/xai
/companies/deepseek
/companies/qwen
/companies/compare
/interview
/interview/coding-set
/how-to-study
/glossary
/math-primer
/paper-reading-protocol
/tech-writing
/search
```

If the script auto-discovers from the sitemap, just add a check that sitemap includes each.

- [ ] **Step 4: Local dry run**

```bash
pnpm --filter book build
pnpm run check-links
```

Expected: exits 0 with each of the routes above reporting 200.

- [ ] **Step 5: Commit**

```bash
git add .github scripts
git commit -m "ci: run vitest + include new platform routes in link check"
```

---

## Task 16: Final QA + production deploy

**Files:** none new.

**Interfaces:** consumes everything.

- [ ] **Step 1: Full clean build + all test suites locally**

```bash
pnpm install --frozen-lockfile
pnpm --filter @core-ai/viz test
pnpm --filter book run test:unit
pnpm --filter book build
pnpm --filter book test
uv run pytest projects/
```

Every step exits 0.

- [ ] **Step 2: Manual QA checklist (run after `pnpm --filter book preview`)**

Confirm each item, then check the box:

- [ ] `/` — home renders, Week 1 in roadmap
- [ ] `/weeks/week-01-linear-algebra` — CompanyLens block shows all 7 rows in fixed order (OpenAI → Qwen)
- [ ] `/companies` — 7×25 matrix renders, sticky headers work, zero horizontal overflow at 375px
- [ ] `/companies/openai … /companies/qwen` — 7 profile pages, each shows model timeline + loop guide + reading list + seeded angles
- [ ] `/companies/compare` — defaults to all 7 / axis=philosophy; `?companies=openai,anthropic&axis=timeline` narrows correctly
- [ ] `/interview` — dashboard renders (0 solved initially); weak-spot heatmap shows "answer some quiz questions first"
- [ ] Answer 2 questions on Week 1 MicroRecall → refresh `/interview` → "Total problems solved" shows 2
- [ ] `/interview/coding-set` — 20 problems in 5 tag groups
- [ ] `⌘K` on any page opens SearchDialog; typing "SVD" returns at least one result linking to Week 1
- [ ] `/search` — dialog auto-opens
- [ ] `/how-to-study`, `/glossary`, `/math-primer`, `/paper-reading-protocol`, `/tech-writing` — all render under ExtraLayout
- [ ] Dark-mode toggle in top-right works and persists across reloads (localStorage `theme`)
- [ ] `prefers-reduced-motion: reduce` — enable OS-level reduce motion, reload `/motion-probe`, box is instantly visible with no fade
- [ ] Playwright suite green (13+ tests)
- [ ] `pnpm run check-links` — exits 0

- [ ] **Step 3: Promote to production**

```bash
npx vercel --prod
```

- [ ] **Step 4: Tag the milestone**

```bash
git tag -a v0.2.0-platform -m "Plan 2 complete: companies + interview + search + motion + companion routes"
git push --tags
```

- [ ] **Step 5: Final commit (empty, marker)**

```bash
git commit --allow-empty -m "chore: plan 2 complete — platform features layer shipped"
```

---

## What Plan 2 leaves for Plan 3+ (foreshadowed, not built)

- **Weekly content**: Weeks 2–4 (Plan 3) will fill in `/weeks/week-02-*` through `/weeks/week-04-*` MDX + 3 more `projects/week-XX-*` Python packages, each with the full 7-company Lens block using the schema already in place.
- **Additional viz primitives**: `AttentionHeatmap`, `GradientDescentField`, `MatrixVectorProduct`, `ProbabilitySimplex` — added into `packages/viz/` as their owning week needs them (Plans 5-6).
- **Interview drill seeding**: Plans 3+ each contribute 1 tagged drill per week; by Plan 8 the dashboard has 15 seeded drills + ~100 angle-only drills.
- **Weak-spot heatmap deep-linking**: the current heatmap cells show a tooltip; clicking to jump to the owning week is deferred — needs an `ownerWeek` field on `AnswerStats`, which we'll add when Plan 3's first quiz lands.
- **Track selection UI (W22-23)**: Track L / P / R picker becomes real when Plan 8 ships the specialization weeks.

## What Plan 3+ ships

- **Plan 3:** Weeks 2–4 (calculus, probability, python+info theory) + 3 reference projects.
- **Plan 4:** Weeks 5–8 (classical ML).
- **Plan 5:** Weeks 9–12 (deep learning).
- **Plan 6:** Weeks 13–17 (transformers/LLMs) — includes W15a/W15b/W17 with cloud-runbook docs.
- **Plan 7:** Weeks 18–21 (systems + safety).
- **Plan 8:** Weeks 22–25 (specialization + capstone kits + interview prep polish).
