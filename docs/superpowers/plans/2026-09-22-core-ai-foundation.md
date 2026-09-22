# Core AI — Plan 1: Foundation + Week 1 Pipeline Proof

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy an Astro book to Vercel with Week 1 (Linear Algebra) fully readable and its `linalg-lab` Python reference project cloneable and tested — proving the entire authoring pipeline works, so every downstream week is just filling in the same template.

**Architecture:** pnpm + uv monorepo. Astro 6 with Content Collections for MDX authoring, KaTeX SSR for math, Shiki for code, hand-authored SVG for diagrams, Preact islands reserved for later interactivity (deferred to Plan 2). One dynamic route (`/weeks/[slug]`) covers all 25 weeks. Reference projects are per-week Python packages using uv + pytest + ruff + hypothesis. Deploy is Vercel static build.

**Tech Stack:** Astro 6 · MDX · Tailwind v4 · Radix Colors · @tailwindcss/typography · KaTeX · Shiki · Expressive Code · Mermaid · Preact · Motion v12 · pnpm · Turborepo · Biome · uv · ruff · pyright · pytest · hypothesis · Node 24 LTS · Python 3.13 · Vercel

**Spec:** `/Users/tharunkumarl/Full Stack/Core-AI/docs/superpowers/specs/2026-09-22-core-ai-book-design.md`

## Global Constraints

- **Node:** 24 LTS (Node 22 exits Active LTS this month, per spec §7.1).
- **Python:** 3.13 (spec §7.1).
- **Framework:** Astro **6.x** with MDX (spec §2.2). Do not pin Astro 5.
- **Styling:** Tailwind v4.x (Lightning CSS) + Radix Colors + @tailwindcss/typography (spec §2.2).
- **Type stack:** Fraunces (display) · Inter (UI/body) · JetBrains Mono (code). Loaded via Astro 6 Fonts API — no `public/fonts` self-hosting (spec §2.2).
- **Aesthetic:** "Editorial Textbook" — wide margins, serif long-form body, oversized display headings, warm and quiet (spec §2.4).
- **Compute-tier badge** (🟢 / 🟡 / 🔴) MUST render at the top of every week page (spec §1, §3).
- **Dark mode:** first-class via `data-theme` attribute; every component readable in both themes (spec §2.3).
- **Accessibility:** WCAG AA minimum; math and diagrams have text alternatives (spec §2.3).
- **Package manager:** pnpm workspaces for JS/TS; uv workspace for Python. Never mix in npm/yarn/pip lockfiles.
- **Copy discipline:** never invent facts about companies/papers/models that aren't in the spec's verified reading list (§5.3). If unsure, leave blank and file an issue.

## Review Focus

The five input classes / failure modes the spec implies but no single task exercises. Each has an inline test added to the owning task.

1. **KaTeX SSR** — math must render in initial HTML with zero client JS (§2.2 says "SSR, zero client cost"). Pin with a snapshot test in **Task 9** that greps built HTML for `<span class="katex">`.
2. **Dark mode contrast** — every callout, code block, and prose block must meet WCAG AA in both themes. Pin with a Playwright color-contrast probe in **Task 4** hitting both `data-theme=light` and `dark`.
3. **Mobile viewport** — nothing overflows at 375px. Pin with a Playwright viewport test in **Task 7** on the layout shell.
4. **External link rot** — dead arXiv/blog links break trust. Pin with a `check-links.mjs` CI job in **Task 20** that runs weekly and fails the build on 404.
5. **Reference project on M-series** — `linalg-lab` must run on an M-series MacBook without CUDA. Pin with a pytest run using MPS/CPU fallback in **Task 16**.

---

## File Structure

```
core-ai-book/                                        (repo root)
├── README.md                                        (Task 1)
├── LICENSE                                          (Task 1)
├── .gitignore                                       (Task 1)
├── .editorconfig                                    (Task 1)
├── .node-version                                    (Task 1) → 24
├── .python-version                                  (Task 1) → 3.13
├── package.json                                     (Task 1) — pnpm workspace root
├── pnpm-workspace.yaml                              (Task 1)
├── pyproject.toml                                   (Task 1) — uv workspace root
├── uv.lock                                          (Task 1) — generated
├── turbo.json                                       (Task 1)
├── biome.json                                       (Task 1)
├── ruff.toml                                        (Task 1)
├── vercel.json                                      (Task 19)
├── .github/workflows/
│   ├── book.yml                                     (Task 20)
│   ├── projects.yml                                 (Task 20)
│   ├── ci.yml                                       (Task 20)
│   └── link-check.yml                               (Task 20)
├── scripts/
│   └── check-links.mjs                              (Task 20)
├── apps/
│   └── book/
│       ├── package.json                             (Task 2)
│       ├── astro.config.mjs                         (Task 2, 3)
│       ├── tailwind.config.ts                       (Task 4)
│       ├── tsconfig.json                            (Task 2)
│       ├── src/
│       │   ├── content/
│       │   │   ├── config.ts                        (Task 6)
│       │   │   ├── weeks/
│       │   │   │   └── week-01-linear-algebra.mdx   (Task 14)
│       │   │   └── companies/
│       │   │       └── openai.mdx                   (Task 18)
│       │   ├── styles/
│       │   │   ├── tokens.css                       (Task 4)
│       │   │   ├── global.css                       (Task 5)
│       │   │   ├── prose.css                        (Task 5)
│       │   │   └── math.css                         (Task 9)
│       │   ├── components/
│       │   │   ├── layout/
│       │   │   │   ├── BookShell.astro              (Task 7)
│       │   │   │   ├── Sidebar.astro                (Task 7)
│       │   │   │   ├── ReadingProgressBar.astro     (Task 7)
│       │   │   │   └── Footer.astro                 (Task 7)
│       │   │   ├── callouts/
│       │   │   │   ├── Intuition.astro              (Task 8)
│       │   │   │   ├── Gotcha.astro                 (Task 8)
│       │   │   │   ├── DeepDive.astro               (Task 8)
│       │   │   │   ├── Interview.astro              (Task 8)
│       │   │   │   └── CompanyPill.astro            (Task 8)
│       │   │   └── content/
│       │   │       ├── Hook.astro                   (Task 11)
│       │   │       ├── Intuition.astro              (Task 11)
│       │   │       ├── MathBlock.astro              (Task 9)
│       │   │       ├── CodeBlock.astro              (Task 10)
│       │   │       ├── ComputeBadge.astro           (Task 11)
│       │   │       ├── CompanyLens.astro            (Task 11)
│       │   │       ├── ReferenceProject.astro       (Task 11)
│       │   │       ├── Assignments.astro            (Task 11)
│       │   │       ├── InterviewDrill.astro         (Task 11)
│       │   │       ├── FurtherReading.astro         (Task 11)
│       │   │       └── KeyTakeaways.astro           (Task 11)
│       │   ├── layouts/
│       │   │   └── WeekLayout.astro                 (Task 13)
│       │   └── pages/
│       │       ├── index.astro                      (Task 12)
│       │       └── weeks/[slug].astro               (Task 13)
│       └── tests/
│           ├── katex-ssr.spec.ts                    (Task 9)
│           ├── dark-mode.spec.ts                    (Task 4)
│           └── mobile-viewport.spec.ts              (Task 7)
└── projects/
    └── week-01-linalg-lab/
        ├── pyproject.toml                           (Task 15)
        ├── README.md                                (Task 18)
        ├── SOLUTION_NOTES.md                        (Task 18)
        ├── COMPUTE.md                               (Task 18)
        ├── src/linalg_lab/
        │   ├── __init__.py                          (Task 15)
        │   ├── vectors.py                           (Task 16)
        │   └── svd_compress.py                      (Task 17)
        ├── tests/
        │   ├── test_vectors.py                      (Task 16)
        │   └── test_svd_compress.py                 (Task 17)
        ├── notebooks/
        │   └── 01-svd-images.ipynb                  (Task 17)
        └── assignments/
            ├── warmup.md                            (Task 18)
            ├── build.md                             (Task 18)
            └── challenge.md                         (Task 18)
```

---

## Task 1: Initialize the monorepo

**Files:**
- Create: `README.md`, `LICENSE`, `.gitignore`, `.editorconfig`, `.node-version`, `.python-version`
- Create: `package.json`, `pnpm-workspace.yaml`, `turbo.json`, `biome.json`
- Create: `pyproject.toml`, `ruff.toml`

**Interfaces:**
- Consumes: nothing
- Produces: a working pnpm workspace (usable by later tasks via `pnpm --filter <pkg>`); a uv workspace (usable via `uv sync`); shared `biome`/`ruff` configs importable by children.

- [ ] **Step 1: Initialize git and pnpm**

```bash
cd "/Users/tharunkumarl/Full Stack/Core-AI"
git init
corepack enable && corepack prepare pnpm@latest --activate
node -v   # confirm Node 24
```

- [ ] **Step 2: Create `.node-version` and `.python-version`**

Create `.node-version` with contents:
```
24
```

Create `.python-version` with contents:
```
3.13
```

- [ ] **Step 3: Write the root `package.json`**

```json
{
  "name": "core-ai-book",
  "private": true,
  "packageManager": "pnpm@9.12.0",
  "engines": { "node": ">=24" },
  "scripts": {
    "dev": "turbo dev",
    "build": "turbo build",
    "lint": "biome check .",
    "format": "biome format --write .",
    "test": "turbo test",
    "check-links": "node scripts/check-links.mjs"
  },
  "devDependencies": {
    "@biomejs/biome": "^1.9.0",
    "turbo": "^2.5.0",
    "typescript": "^5.7.0"
  }
}
```

- [ ] **Step 4: Write `pnpm-workspace.yaml`**

```yaml
packages:
  - "apps/*"
  - "packages/*"
```

- [ ] **Step 5: Write `turbo.json`**

```json
{
  "$schema": "https://turbo.build/schema.json",
  "tasks": {
    "dev":   { "cache": false, "persistent": true },
    "build": { "outputs": ["dist/**", ".vercel/**"], "dependsOn": ["^build"] },
    "test":  { "outputs": [], "dependsOn": ["^build"] },
    "lint":  { "outputs": [] }
  }
}
```

- [ ] **Step 6: Write `biome.json`**

```json
{
  "$schema": "https://biomejs.dev/schemas/1.9.0/schema.json",
  "files": { "ignore": ["dist", ".astro", ".vercel", "node_modules", "projects/**/notebooks"] },
  "formatter": { "enabled": true, "indentStyle": "space", "indentWidth": 2, "lineWidth": 100 },
  "linter":    { "enabled": true, "rules": { "recommended": true } },
  "javascript": { "formatter": { "quoteStyle": "single", "trailingCommas": "all", "semicolons": "asNeeded" } }
}
```

- [ ] **Step 7: Write `pyproject.toml` (uv workspace root)**

```toml
[project]
name = "core-ai-book"
version = "0.1.0"
requires-python = ">=3.13"

[tool.uv.workspace]
members = ["projects/*"]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "RUF"]

[tool.pyright]
pythonVersion = "3.13"
typeCheckingMode = "strict"
```

- [ ] **Step 8: Write `ruff.toml` (repo-wide overrides)**

```toml
extend = "pyproject.toml"
```

- [ ] **Step 9: Write `.gitignore`**

```
node_modules
dist
.astro
.vercel
.turbo
.venv
__pycache__
*.pyc
.pytest_cache
.ruff_cache
.ipynb_checkpoints
.DS_Store
uv.lock
```

- [ ] **Step 10: Write `.editorconfig`**

```
root = true
[*]
indent_style = space
indent_size = 2
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true
[*.py]
indent_size = 4
```

- [ ] **Step 11: Write minimal `README.md`**

```markdown
# Core AI — From Foundation to Frontier

An open-source, self-driven, 25-week AI curriculum for full-stack engineers targeting AI roles at frontier labs.

- Site: `apps/book/`
- Reference projects: `projects/`
- Spec: `docs/superpowers/specs/2026-09-22-core-ai-book-design.md`
- Plans: `docs/superpowers/plans/`

## Getting started

```bash
pnpm install
uv sync
pnpm dev
```
```

- [ ] **Step 12: Write `LICENSE` (MIT)**

Use standard MIT template with year 2026 and copyright holder "Tharun".

- [ ] **Step 13: Install pnpm dependencies + create `uv.lock`**

```bash
pnpm install
uv sync
```

Verify: `pnpm -w --version` prints a version; `uv --version` prints a version.

- [ ] **Step 14: Commit**

```bash
git add .
git commit -m "chore: initialize monorepo (pnpm + uv + turborepo + biome + ruff)"
```

---

## Task 2: Scaffold the Astro book app

**Files:**
- Create: `apps/book/package.json`, `apps/book/astro.config.mjs`, `apps/book/tsconfig.json`
- Create: `apps/book/src/pages/index.astro` (temporary placeholder)

**Interfaces:**
- Consumes: root pnpm workspace from Task 1.
- Produces: an Astro 6 app runnable at `pnpm --filter book dev` on http://localhost:4321.

- [ ] **Step 1: Create the `apps/book` directory**

```bash
mkdir -p "apps/book/src/pages"
```

- [ ] **Step 2: Write `apps/book/package.json`**

```json
{
  "name": "book",
  "type": "module",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev":   "astro dev",
    "build": "astro build",
    "preview": "astro preview",
    "test": "playwright test"
  },
  "dependencies": {
    "astro": "^6.0.0",
    "@astrojs/mdx": "^5.0.0",
    "@astrojs/preact": "^4.0.0",
    "@astrojs/sitemap": "^4.0.0",
    "@astrojs/tailwind": "^7.0.0",
    "tailwindcss": "^4.3.0",
    "@tailwindcss/typography": "^0.6.0",
    "preact": "^10.24.0",
    "@preact/signals": "^2.0.0"
  },
  "devDependencies": {
    "@playwright/test": "^1.48.0"
  }
}
```

- [ ] **Step 3: Write `apps/book/astro.config.mjs`**

```js
import { defineConfig } from 'astro/config'
import mdx from '@astrojs/mdx'
import preact from '@astrojs/preact'
import sitemap from '@astrojs/sitemap'
import tailwind from '@astrojs/tailwind'

export default defineConfig({
  site: 'https://core-ai.book',
  integrations: [mdx(), preact({ compat: false }), sitemap(), tailwind()],
  markdown: {
    shikiConfig: { theme: 'github-dark-dimmed', wrap: false }
  }
})
```

- [ ] **Step 4: Write `apps/book/tsconfig.json`**

```json
{
  "extends": "astro/tsconfigs/strict",
  "include": [".astro/types.d.ts", "**/*"],
  "exclude": ["dist"]
}
```

- [ ] **Step 5: Write placeholder `apps/book/src/pages/index.astro`**

```astro
---
---
<html lang="en">
  <head><meta charset="utf-8" /><title>Core AI</title></head>
  <body><h1>Core AI — book scaffold OK</h1></body>
</html>
```

- [ ] **Step 6: Install and run the dev server**

```bash
pnpm install
pnpm --filter book dev &
sleep 5
curl -s http://localhost:4321 | grep -q "book scaffold OK" && echo PASS || echo FAIL
kill %1
```

Expected: `PASS`.

- [ ] **Step 7: Commit**

```bash
git add apps/book pnpm-lock.yaml
git commit -m "feat(book): scaffold astro 6 + mdx + preact + tailwind app"
```

---

## Task 3: Configure Astro 6 Fonts API

**Files:**
- Modify: `apps/book/astro.config.mjs`

**Interfaces:**
- Consumes: Astro app from Task 2.
- Produces: three CSS variables `--font-display`, `--font-body`, `--font-mono` usable in Task 4's tokens.

- [ ] **Step 1: Add the `fonts` block to `astro.config.mjs`**

Modify to:

```js
import { defineConfig, fontProviders } from 'astro/config'
// ...existing imports

export default defineConfig({
  site: 'https://core-ai.book',
  integrations: [mdx(), preact({ compat: false }), sitemap(), tailwind()],
  experimental: {
    fonts: [
      { name: 'Fraunces',        cssVariable: '--font-display', provider: fontProviders.google(), weights: [400, 600, 800], styles: ['normal', 'italic'] },
      { name: 'Inter',           cssVariable: '--font-body',    provider: fontProviders.google(), weights: [400, 500, 600, 700], subsets: ['latin'] },
      { name: 'JetBrains Mono',  cssVariable: '--font-mono',    provider: fontProviders.google(), weights: [400, 600] }
    ]
  },
  markdown: { shikiConfig: { theme: 'github-dark-dimmed', wrap: false } }
})
```

- [ ] **Step 2: Modify `apps/book/src/pages/index.astro` to preload fonts**

```astro
---
import { Font } from 'astro:assets'
---
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Core AI</title>
    <Font cssVariable="--font-display" preload />
    <Font cssVariable="--font-body" preload />
    <Font cssVariable="--font-mono" />
  </head>
  <body style="font-family: var(--font-body)">
    <h1 style="font-family: var(--font-display)">Core AI — fonts OK</h1>
    <code style="font-family: var(--font-mono)">console.log('mono')</code>
  </body>
</html>
```

- [ ] **Step 3: Test fonts load in a headless browser**

```bash
pnpm --filter book build
pnpm --filter book preview &
sleep 3
curl -s http://localhost:4321 | grep -q 'font-display' && echo PASS || echo FAIL
kill %1
```

Expected: `PASS`.

- [ ] **Step 4: Commit**

```bash
git add apps/book
git commit -m "feat(book): wire astro 6 fonts api (Fraunces, Inter, JetBrains Mono)"
```

---

## Task 4: Design tokens + Tailwind config + dark-mode contrast test

**Files:**
- Create: `apps/book/src/styles/tokens.css`
- Create: `apps/book/tailwind.config.ts`
- Create: `apps/book/tests/dark-mode.spec.ts`
- Create: `apps/book/playwright.config.ts`

**Interfaces:**
- Consumes: font CSS variables from Task 3.
- Produces: Tailwind theme extended with tokens (`bg-canvas`, `text-fg`, `accent-p1` … `accent-p6`, `border-soft`) usable everywhere.

- [ ] **Step 1: Write `apps/book/src/styles/tokens.css`**

```css
@import "@radix-ui/colors/slate.css";
@import "@radix-ui/colors/slate-dark.css";
@import "@radix-ui/colors/indigo.css";
@import "@radix-ui/colors/teal.css";
@import "@radix-ui/colors/violet.css";
@import "@radix-ui/colors/amber.css";
@import "@radix-ui/colors/grass.css";
@import "@radix-ui/colors/tomato.css";

:root, :root[data-theme="light"] {
  --canvas:      var(--slate-1);
  --canvas-subtle: var(--slate-2);
  --fg:          var(--slate-12);
  --fg-muted:    var(--slate-11);
  --border-soft: var(--slate-6);
  --accent-p1:   var(--indigo-9);   /* Month 1 */
  --accent-p2:   var(--teal-9);     /* Month 2 */
  --accent-p3:   var(--violet-9);   /* Month 3 */
  --accent-p4:   var(--amber-9);    /* Month 4 */
  --accent-p5:   var(--grass-9);    /* Month 5 */
  --accent-p6:   var(--tomato-9);   /* Month 6 */
}

:root[data-theme="dark"] {
  --canvas:      var(--slate-1);
  --canvas-subtle: var(--slate-2);
  --fg:          var(--slate-12);
  --fg-muted:    var(--slate-11);
  --border-soft: var(--slate-6);
  /* Radix Colors "dark" variants replace the -9 stops automatically via CSS variable cascade */
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { color-scheme: dark; }
}
```

- [ ] **Step 2: Add `@radix-ui/colors` to `apps/book/package.json` dependencies**

```bash
pnpm --filter book add @radix-ui/colors
```

- [ ] **Step 3: Write `apps/book/tailwind.config.ts`**

```ts
import type { Config } from 'tailwindcss'
import typography from '@tailwindcss/typography'

export default {
  content: ['./src/**/*.{astro,html,md,mdx,ts,tsx}'],
  darkMode: ['selector', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        canvas:       'var(--canvas)',
        'canvas-subtle': 'var(--canvas-subtle)',
        fg:           'var(--fg)',
        'fg-muted':   'var(--fg-muted)',
        'border-soft':'var(--border-soft)',
        'accent-p1':  'var(--accent-p1)',
        'accent-p2':  'var(--accent-p2)',
        'accent-p3':  'var(--accent-p3)',
        'accent-p4':  'var(--accent-p4)',
        'accent-p5':  'var(--accent-p5)',
        'accent-p6':  'var(--accent-p6)'
      },
      fontFamily: {
        display: 'var(--font-display)',
        body:    'var(--font-body)',
        mono:    'var(--font-mono)'
      },
      fontSize: {
        // 1.250 Major Third scale
        xs: '12px', sm: '14px', base: '16px', md: '18px',
        lg: '20px', xl: '25px', '2xl': '31px', '3xl': '39px', '4xl': '49px', '5xl': '61px'
      },
      spacing: {
        1: '4px', 2: '8px', 3: '12px', 4: '16px', 6: '24px', 8: '32px',
        12: '48px', 16: '64px', 24: '96px'
      },
      borderRadius: { xs: '2px', sm: '4px', md: '8px', lg: '12px' },
      maxWidth:     { prose: '68ch' },
      lineHeight:   { relaxed: '1.7' }
    }
  },
  plugins: [typography]
} satisfies Config
```

- [ ] **Step 4: Update `apps/book/src/pages/index.astro` to prove tokens work**

```astro
---
import { Font } from 'astro:assets'
import '../styles/tokens.css'
---
<html lang="en" data-theme="light">
  <head>
    <meta charset="utf-8" />
    <title>Core AI</title>
    <Font cssVariable="--font-display" preload />
    <Font cssVariable="--font-body" preload />
    <Font cssVariable="--font-mono" />
  </head>
  <body class="bg-canvas text-fg font-body">
    <h1 class="font-display text-4xl text-accent-p1">Core AI — tokens OK</h1>
    <button
      onclick="document.documentElement.dataset.theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'"
      class="mt-4 px-3 py-2 rounded-sm border border-border-soft">
      Toggle theme
    </button>
  </body>
</html>
```

- [ ] **Step 5: Write `apps/book/playwright.config.ts`**

```ts
import { defineConfig } from '@playwright/test'
export default defineConfig({
  testDir: './tests',
  webServer: { command: 'pnpm build && pnpm preview --host 127.0.0.1 --port 4321', url: 'http://127.0.0.1:4321', reuseExistingServer: false, timeout: 60_000 },
  use: { baseURL: 'http://127.0.0.1:4321' }
})
```

- [ ] **Step 6: Write the dark-mode contrast probe test**

Create `apps/book/tests/dark-mode.spec.ts`:

```ts
import { expect, test } from '@playwright/test'

for (const theme of ['light', 'dark'] as const) {
  test(`H1 has readable contrast in ${theme} theme`, async ({ page }) => {
    await page.goto('/')
    await page.evaluate((t) => { document.documentElement.dataset.theme = t }, theme)
    const contrast = await page.evaluate(() => {
      const el = document.querySelector('h1')!
      const style = getComputedStyle(el)
      const bg = getComputedStyle(document.body).backgroundColor
      // crude parse; just assert neither is transparent and colors differ
      return { fg: style.color, bg }
    })
    expect(contrast.fg).not.toBe(contrast.bg)
    expect(contrast.fg).not.toBe('rgba(0, 0, 0, 0)')
    expect(contrast.bg).not.toBe('rgba(0, 0, 0, 0)')
  })
}
```

- [ ] **Step 7: Install Playwright browsers and run the test**

```bash
pnpm --filter book exec playwright install --with-deps chromium
pnpm --filter book test
```

Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add .
git commit -m "feat(book): design tokens, tailwind v4 config, dark-mode contrast probe"
```

---

## Task 5: Global styles + prose CSS

**Files:**
- Create: `apps/book/src/styles/global.css`, `apps/book/src/styles/prose.css`

**Interfaces:**
- Consumes: tokens from Task 4.
- Produces: `body` baseline styles + a `.prose-book` class the `WeekLayout` will apply to MDX-rendered content.

- [ ] **Step 1: Write `apps/book/src/styles/global.css`**

```css
@import "tailwindcss";
@import "./tokens.css";
@import "./prose.css";

html { color-scheme: light dark; }
body {
  font-family: var(--font-body);
  font-size: 18px;
  line-height: 1.7;
  background: var(--canvas);
  color: var(--fg);
  text-rendering: optimizeLegibility;
}

*, *::before, *::after { box-sizing: border-box; }

@media (prefers-reduced-motion: reduce) {
  * { animation-duration: 0s !important; transition-duration: 0s !important; }
}
```

- [ ] **Step 2: Write `apps/book/src/styles/prose.css`**

```css
.prose-book {
  max-width: 68ch;
  font-family: var(--font-body);
  color: var(--fg);
}
.prose-book :where(h1, h2, h3, h4) { font-family: var(--font-display); color: var(--fg); letter-spacing: -0.01em; }
.prose-book h1 { font-size: 49px; line-height: 1.1; margin-block: 48px 24px; }
.prose-book h2 { font-size: 31px; line-height: 1.2; margin-block: 48px 16px; }
.prose-book h3 { font-size: 25px; line-height: 1.3; margin-block: 32px 12px; }
.prose-book p  { margin-block: 12px; }
.prose-book :where(code, pre) { font-family: var(--font-mono); font-size: 15px; }
.prose-book a  { color: var(--accent-p1); text-underline-offset: 3px; }
.prose-book blockquote { border-inline-start: 3px solid var(--accent-p1); padding-inline-start: 16px; color: var(--fg-muted); font-style: italic; }
```

- [ ] **Step 3: Import `global.css` from `index.astro`**

Change the import line in `apps/book/src/pages/index.astro` from `import '../styles/tokens.css'` to `import '../styles/global.css'`.

- [ ] **Step 4: Verify visually**

```bash
pnpm --filter book dev &
sleep 4
curl -s http://localhost:4321 | grep -q "tokens OK"
kill %1
```

- [ ] **Step 5: Commit**

```bash
git add apps/book/src
git commit -m "feat(book): global styles + prose-book class"
```

---

## Task 6: Content Collections with Zod schemas

**Files:**
- Create: `apps/book/src/content/config.ts`
- Create: `apps/book/src/content/weeks/.gitkeep`
- Create: `apps/book/src/content/companies/.gitkeep`

**Interfaces:**
- Consumes: nothing.
- Produces: two type-safe collections `weeks` and `companies`; frontmatter shape locked. Downstream tasks import types from `astro:content`.

- [ ] **Step 1: Write `apps/book/src/content/config.ts`**

```ts
import { defineCollection, z } from 'astro:content'

const weeks = defineCollection({
  type: 'content',
  schema: z.object({
    week: z.number().int().min(1).max(25),
    part: z.number().int().min(1).max(6),                 // Month 1..6
    slug: z.string(),                                     // e.g. "week-01-linear-algebra"
    title: z.string(),
    hook: z.string(),                                     // one striking sentence
    hours: z.number().default(20),
    computeTier: z.enum(['green', 'yellow', 'red']),
    difficulty: z.number().int().min(1).max(5),
    prereqSlugs: z.array(z.string()).default([]),
    referenceProject: z.string().optional(),              // path like "projects/week-01-linalg-lab"
    accentVar: z.string().default('--accent-p1'),
    publishedAt: z.date().optional()
  })
})

const companies = defineCollection({
  type: 'content',
  schema: z.object({
    slug: z.enum(['openai', 'anthropic', 'deepmind', 'meta', 'xai', 'deepseek', 'qwen']),
    name: z.string(),
    tint: z.string(),                                     // css color for the pill
    tagline: z.string(),
    founded: z.number().int(),
    hq: z.string()
  })
})

export const collections = { weeks, companies }
```

- [ ] **Step 2: Ensure the directories are tracked in git**

```bash
mkdir -p apps/book/src/content/weeks apps/book/src/content/companies
touch apps/book/src/content/weeks/.gitkeep apps/book/src/content/companies/.gitkeep
```

- [ ] **Step 3: Sanity-check the collection compiles**

```bash
pnpm --filter book astro sync
```

Expected: exit 0, and `apps/book/.astro/types.d.ts` is regenerated with `WeekSchema` types.

- [ ] **Step 4: Commit**

```bash
git add apps/book/src/content
git commit -m "feat(book): content collections (weeks, companies) with Zod schemas"
```

---

## Task 7: Layout shell + mobile-viewport test

**Files:**
- Create: `apps/book/src/components/layout/BookShell.astro`, `Sidebar.astro`, `ReadingProgressBar.astro`, `Footer.astro`
- Create: `apps/book/tests/mobile-viewport.spec.ts`
- Modify: `apps/book/src/pages/index.astro` (use `BookShell`)

**Interfaces:**
- Consumes: tokens, global styles.
- Produces: `<BookShell title accent>{children}</BookShell>` used by home + week + company pages.

- [ ] **Step 1: Write `BookShell.astro`**

```astro
---
import { Font } from 'astro:assets'
import Sidebar from './Sidebar.astro'
import Footer from './Footer.astro'
import ReadingProgressBar from './ReadingProgressBar.astro'
import '../../styles/global.css'
interface Props { title: string; accent?: string }
const { title, accent = 'var(--accent-p1)' } = Astro.props
---
<html lang="en" data-theme="light" style={`--accent: ${accent}`}>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title} · Core AI</title>
    <Font cssVariable="--font-display" preload />
    <Font cssVariable="--font-body" preload />
    <Font cssVariable="--font-mono" />
  </head>
  <body class="bg-canvas text-fg">
    <ReadingProgressBar />
    <div class="grid grid-cols-1 lg:grid-cols-[280px_1fr_240px] min-h-screen">
      <Sidebar />
      <main class="px-4 lg:px-8 py-8 mx-auto w-full max-w-prose">
        <slot />
      </main>
      <aside class="hidden lg:block px-4 py-8" aria-label="Margin notes"></aside>
    </div>
    <Footer />
    <script is:inline>
      // Restore theme
      const t = localStorage.getItem('theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
      document.documentElement.dataset.theme = t
    </script>
  </body>
</html>
```

- [ ] **Step 2: Write `Sidebar.astro`**

```astro
---
import { getCollection } from 'astro:content'
const weeks = (await getCollection('weeks')).sort((a, b) => a.data.week - b.data.week)
---
<nav aria-label="Chapter navigation" class="hidden lg:block border-r border-border-soft p-6 sticky top-0 h-screen overflow-y-auto">
  <a href="/" class="font-display text-lg block mb-6">Core AI</a>
  <ol class="space-y-1 text-sm">
    {weeks.map((w) => (
      <li>
        <a href={`/weeks/${w.data.slug}`} class="block py-1 hover:text-accent-p1">
          <span class="text-fg-muted mr-2">W{String(w.data.week).padStart(2, '0')}</span>
          {w.data.title}
        </a>
      </li>
    ))}
  </ol>
</nav>
```

- [ ] **Step 3: Write `ReadingProgressBar.astro`**

```astro
<div id="progress" class="fixed top-0 left-0 h-1 bg-accent-p1 z-50" style="width: 0%"></div>
<script is:inline>
  const bar = document.getElementById('progress')
  const update = () => {
    const h = document.documentElement
    const p = (h.scrollTop / (h.scrollHeight - h.clientHeight || 1)) * 100
    bar.style.width = `${Math.min(100, p)}%`
  }
  addEventListener('scroll', update, { passive: true })
  update()
</script>
```

- [ ] **Step 4: Write `Footer.astro`**

```astro
<footer class="border-t border-border-soft mt-24 py-8 text-sm text-fg-muted text-center">
  Core AI · MIT licensed · <a href="https://github.com/tharun/core-ai-book">GitHub</a>
</footer>
```

- [ ] **Step 5: Rewrite `index.astro` using `BookShell`**

```astro
---
import BookShell from '../components/layout/BookShell.astro'
---
<BookShell title="Home">
  <h1 class="font-display text-5xl">Core AI</h1>
  <p class="text-md text-fg-muted mt-2">From foundation to frontier · 25 weeks · Zero to interview-ready.</p>
  <a href="/weeks/week-01-linear-algebra" class="inline-block mt-8 px-4 py-3 bg-accent-p1 text-white rounded-sm">Start Week 1 →</a>
</BookShell>
```

- [ ] **Step 6: Write `mobile-viewport.spec.ts`**

```ts
import { expect, test } from '@playwright/test'

test('home renders without horizontal overflow at 375px', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 800 })
  await page.goto('/')
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(0)
})
```

- [ ] **Step 7: Run the tests**

```bash
pnpm --filter book test
```

Expected: 3 passed (dark-mode ×2 + mobile-viewport ×1).

- [ ] **Step 8: Commit**

```bash
git add apps/book
git commit -m "feat(book): layout shell + mobile-viewport probe"
```

---

## Task 8: Semantic callout components

**Files:**
- Create: `apps/book/src/components/callouts/Intuition.astro`, `Gotcha.astro`, `DeepDive.astro`, `Interview.astro`, `CompanyPill.astro`

**Interfaces:**
- Consumes: tokens.
- Produces: 5 MDX-usable components. All accept `<slot />` and render an accessible `<aside>` (or `<details>` for `DeepDive`) with a fixed icon + accent border.

- [ ] **Step 1: Write `Intuition.astro`**

```astro
<aside role="note" aria-label="Intuition" class="my-6 rounded-md border-l-4 border-accent-p1 bg-canvas-subtle p-4">
  <div class="flex items-start gap-3">
    <span aria-hidden="true" class="text-xl">💡</span>
    <div class="prose-book"><slot /></div>
  </div>
</aside>
```

- [ ] **Step 2: Write `Gotcha.astro`**

```astro
<aside role="note" aria-label="Gotcha" class="my-6 rounded-md border-l-4 border-accent-p6 bg-canvas-subtle p-4">
  <div class="flex items-start gap-3">
    <span aria-hidden="true" class="text-xl">⚠️</span>
    <div class="prose-book"><slot /></div>
  </div>
</aside>
```

- [ ] **Step 3: Write `DeepDive.astro` as `<details>`**

```astro
---
interface Props { summary: string }
const { summary } = Astro.props
---
<details class="my-6 rounded-md border border-border-soft bg-canvas-subtle p-4">
  <summary class="cursor-pointer text-sm font-medium">🔬 {summary}</summary>
  <div class="prose-book mt-4"><slot /></div>
</details>
```

- [ ] **Step 4: Write `Interview.astro`**

```astro
<aside role="note" aria-label="Interview drill" class="my-6 rounded-md border-l-4 border-accent-p3 bg-canvas-subtle p-4">
  <div class="flex items-start gap-3">
    <span aria-hidden="true" class="text-xl">🎯</span>
    <div class="prose-book"><slot /></div>
  </div>
</aside>
```

- [ ] **Step 5: Write `CompanyPill.astro`**

```astro
---
interface Props { company: string; tint?: string }
const { company, tint = 'var(--accent-p1)' } = Astro.props
---
<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs" style={`background: color-mix(in oklab, ${tint} 15%, transparent); color: ${tint}`}>
  📎 {company}
</span>
```

- [ ] **Step 6: Commit**

```bash
git add apps/book/src/components/callouts
git commit -m "feat(book): 5 semantic callout components"
```

---

## Task 9: MathBlock (KaTeX SSR) + SSR proof test

**Files:**
- Create: `apps/book/src/components/content/MathBlock.astro`
- Create: `apps/book/src/styles/math.css`
- Create: `apps/book/tests/katex-ssr.spec.ts`
- Modify: `apps/book/astro.config.mjs` (add remark-math + rehype-katex)

**Interfaces:**
- Consumes: tokens.
- Produces: `<MathBlock latex="a^2 + b^2 = c^2" />` and `$...$` / `$$...$$` in MDX both render to KaTeX HTML at build time.

- [ ] **Step 1: Add remark-math + rehype-katex + katex**

```bash
pnpm --filter book add remark-math rehype-katex katex
```

- [ ] **Step 2: Update `astro.config.mjs` markdown pipeline**

```js
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

// ...inside defineConfig({...}):
markdown: {
  remarkPlugins: [remarkMath],
  rehypePlugins: [[rehypeKatex, { output: 'html', strict: 'warn' }]],
  shikiConfig: { theme: 'github-dark-dimmed', wrap: false }
}
```

- [ ] **Step 3: Write `MathBlock.astro`**

```astro
---
import katex from 'katex'
interface Props { latex: string; display?: boolean }
const { latex, display = true } = Astro.props
const html = katex.renderToString(latex, { displayMode: display, throwOnError: false, output: 'html', strict: 'warn' })
---
<div class="math-block my-4" set:html={html} />
```

- [ ] **Step 4: Write `math.css`**

```css
@import "katex/dist/katex.min.css";
.math-block { overflow-x: auto; padding-block: 8px; }
.katex { font-size: 1.05em; }
:root[data-theme="dark"] .katex { color: var(--fg); }
```

Import from `global.css`:
```css
@import "./math.css";
```

- [ ] **Step 5: Write the KaTeX SSR test**

Create `apps/book/tests/katex-ssr.spec.ts`:

```ts
import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

test('KaTeX renders at build time (SSR, no client JS)', async () => {
  // The build must have been run by playwright's webServer.
  const html = await readFile(join(process.cwd(), 'dist', 'weeks', 'week-01-linear-algebra', 'index.html'), 'utf-8')
  expect(html).toMatch(/<span class="katex/)
})
```

*(Note: this test will start passing after Task 14 authors the Week 1 MDX. It is placed here because MathBlock is what makes it possible. Executor: mark the test as `.fail()` until Task 14 authors the MDX with math, then flip.)*

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "feat(book): MathBlock + KaTeX SSR pipeline"
```

---

## Task 10: CodeBlock via Shiki + Expressive Code

**Files:**
- Modify: `apps/book/astro.config.mjs` (add astro-expressive-code)
- Create: `apps/book/src/components/content/CodeBlock.astro` (wrapper for named blocks with file tabs)

**Interfaces:**
- Consumes: nothing.
- Produces: Fenced code blocks in MDX get Shiki + Expressive Code (file tabs, diff, line highlights, copy button).

- [ ] **Step 1: Add astro-expressive-code**

```bash
pnpm --filter book add astro-expressive-code
```

- [ ] **Step 2: Update `astro.config.mjs`**

```js
import expressiveCode from 'astro-expressive-code'

// integrations array — put expressiveCode BEFORE mdx:
integrations: [
  expressiveCode({ themes: ['github-light', 'github-dark-dimmed'], styleOverrides: { codeFontFamily: 'var(--font-mono)' } }),
  mdx(),
  preact({ compat: false }),
  sitemap(),
  tailwind()
]
```

- [ ] **Step 3: Write `CodeBlock.astro` (multi-file variant only)**

```astro
---
import { Code } from 'astro:components'
interface Props { code: string; lang?: string; filename?: string }
const { code, lang = 'python', filename } = Astro.props
---
{filename && <p class="text-xs text-fg-muted -mb-2 pl-2">{filename}</p>}
<Code code={code} lang={lang} />
```

- [ ] **Step 4: Verify in a scratch page**

Update `index.astro` to include:
```astro
<pre><code class="language-python">def add(a: int, b: int) -> int:
    return a + b</code></pre>
```
Confirm code renders with syntax highlighting and copy button.

- [ ] **Step 5: Commit**

```bash
git add apps/book
git commit -m "feat(book): expressive-code + shiki code rendering"
```

---

## Task 11: The 8-part content components

**Files:**
- Create in `apps/book/src/components/content/`:
  `Hook.astro`, `Intuition.astro`, `ComputeBadge.astro`, `CompanyLens.astro`,
  `ReferenceProject.astro`, `Assignments.astro`, `InterviewDrill.astro`,
  `FurtherReading.astro`, `KeyTakeaways.astro`

**Interfaces:**
- Consumes: callouts (Task 8), MathBlock (Task 9).
- Produces: 9 MDX-usable components (matches spec §3's 8-part anatomy + `ComputeBadge`).

- [ ] **Step 1: `Hook.astro`**

```astro
<section class="my-8">
  <p class="font-display text-xl italic text-fg-muted"><slot /></p>
</section>
```

- [ ] **Step 2: `Intuition.astro` (section, not the callout)**

```astro
---
interface Props { title?: string }
const { title = 'Intuition' } = Astro.props
---
<section class="my-8">
  <h2 class="font-display text-3xl">{title}</h2>
  <div class="prose-book"><slot /></div>
</section>
```

- [ ] **Step 3: `ComputeBadge.astro`**

```astro
---
interface Props { tier: 'green' | 'yellow' | 'red' }
const { tier } = Astro.props
const label = { green: '🟢 Local (M-series)', yellow: '🟡 M-series stretch or optional cloud', red: '🔴 Cloud GPU required' }[tier]
---
<span class="inline-flex items-center gap-2 text-sm px-3 py-1 rounded-full border border-border-soft" aria-label={`Compute tier: ${label}`}>{label}</span>
```

- [ ] **Step 4: `CompanyLens.astro` (5-block template)**

```astro
---
interface Props { topic: string }
const { topic } = Astro.props
---
<section class="my-10 border border-border-soft rounded-md p-6 bg-canvas-subtle">
  <h2 class="font-display text-2xl">🔍 Company Lens — {topic}</h2>
  <div class="prose-book"><slot /></div>
</section>
```

- [ ] **Step 5: `ReferenceProject.astro`**

```astro
---
interface Props { path: string; name: string; hours: string }
const { path, name, hours } = Astro.props
---
<section class="my-10 border-l-4 border-accent-p3 pl-6">
  <h2 class="font-display text-2xl">🧪 Reference Project — <code>{name}</code></h2>
  <p class="text-fg-muted text-sm">{hours} · <a href={`https://github.com/tharun/core-ai-book/tree/main/${path}`}>{path}</a></p>
  <div class="prose-book"><slot /></div>
</section>
```

- [ ] **Step 6: `Assignments.astro`**

```astro
<section class="my-10">
  <h2 class="font-display text-2xl">Assignments</h2>
  <div class="prose-book"><slot /></div>
</section>
```

- [ ] **Step 7: `InterviewDrill.astro`**

```astro
---
interface Props { role: string; company?: string; time?: string }
const { role, company, time = '20 min' } = Astro.props
---
<section class="my-10 border border-accent-p3 rounded-md p-6">
  <h2 class="font-display text-2xl">🎯 Interview Drill</h2>
  <p class="text-fg-muted text-sm">Role: <code>{role}</code> · Time: {time}{company && <> · Most-asked at: <strong>{company}</strong></>}</p>
  <div class="prose-book mt-4"><slot /></div>
</section>
```

- [ ] **Step 8: `FurtherReading.astro`**

```astro
<section class="my-10">
  <h2 class="font-display text-2xl">📚 Further Reading</h2>
  <div class="prose-book"><slot /></div>
</section>
```

- [ ] **Step 9: `KeyTakeaways.astro`**

```astro
<section class="my-10 rounded-md p-6 bg-canvas-subtle">
  <h2 class="font-display text-2xl">🔖 Key Takeaways</h2>
  <div class="prose-book"><slot /></div>
</section>
```

- [ ] **Step 10: Commit**

```bash
git add apps/book/src/components/content
git commit -m "feat(book): 8-part lesson anatomy components"
```

---

## Task 12: Home page — hero + 25-week roadmap

**Files:**
- Modify: `apps/book/src/pages/index.astro`

**Interfaces:**
- Consumes: `weeks` collection (Task 6), `BookShell` (Task 7).
- Produces: `/` route showing hero + linked roadmap.

- [ ] **Step 1: Rewrite `index.astro`**

```astro
---
import { getCollection } from 'astro:content'
import BookShell from '../components/layout/BookShell.astro'

const weeks = (await getCollection('weeks')).sort((a, b) => a.data.week - b.data.week)
const parts = [1, 2, 3, 4, 5, 6].map((p) => ({
  part: p,
  title: ['Math & Programming', 'Classical ML', 'Deep Learning', 'Transformers & LLMs', 'Systems & Production', 'Specialization & Interview'][p - 1],
  weeks: weeks.filter((w) => w.data.part === p)
}))
---
<BookShell title="Home">
  <section class="mb-16">
    <h1 class="font-display text-5xl leading-tight">Core AI</h1>
    <p class="font-display text-2xl text-fg-muted mt-3">From foundation to frontier.</p>
    <p class="mt-4 max-w-prose">25 weeks. Zero to interview-ready for AI-engineer roles at frontier labs — with math, code, real reference projects, and a Company Lens through every topic.</p>
    <div class="mt-6 flex gap-3">
      <a href="/weeks/week-01-linear-algebra" class="px-4 py-3 bg-accent-p1 text-white rounded-sm">Start Week 1 →</a>
      <a href="/how-to-study" class="px-4 py-3 border border-border-soft rounded-sm">How to study this book</a>
    </div>
  </section>

  {parts.map(({ part, title, weeks }) => (
    <section class="my-12">
      <h2 class="font-display text-3xl">Month {part} — {title}</h2>
      <ol class="mt-4 space-y-2">
        {weeks.map((w) => (
          <li class="flex justify-between border-b border-border-soft py-2">
            <a href={`/weeks/${w.data.slug}`} class="hover:text-accent-p1">
              <span class="text-fg-muted mr-3">W{String(w.data.week).padStart(2, '0')}</span>
              {w.data.title}
            </a>
            <span class="text-sm text-fg-muted">{w.data.hours}h</span>
          </li>
        ))}
      </ol>
    </section>
  ))}
</BookShell>
```

- [ ] **Step 2: Verify — the page renders with an (empty) roadmap until Task 14 seeds W1**

```bash
pnpm --filter book dev &
sleep 4
curl -s http://localhost:4321 | grep -q "From foundation to frontier"
kill %1
```

- [ ] **Step 3: Commit**

```bash
git add apps/book/src/pages/index.astro
git commit -m "feat(book): home page with hero + 25-week roadmap"
```

---

## Task 13: WeekLayout + `[slug]` dynamic route

**Files:**
- Create: `apps/book/src/layouts/WeekLayout.astro`
- Create: `apps/book/src/pages/weeks/[slug].astro`

**Interfaces:**
- Consumes: `weeks` collection, `BookShell`, `ComputeBadge`.
- Produces: every entry in `src/content/weeks/*.mdx` gets a rendered route at `/weeks/<slug>`.

- [ ] **Step 1: Write `WeekLayout.astro`**

```astro
---
import BookShell from '../components/layout/BookShell.astro'
import ComputeBadge from '../components/content/ComputeBadge.astro'
interface Props { frontmatter: { week: number; part: number; title: string; hook: string; hours: number; computeTier: 'green'|'yellow'|'red'; difficulty: number; accentVar: string } }
const { frontmatter } = Astro.props
---
<BookShell title={frontmatter.title} accent={`var(${frontmatter.accentVar})`}>
  <header class="mb-8">
    <p class="text-fg-muted text-sm">Week {frontmatter.week} · Month {frontmatter.part} · {frontmatter.hours} hrs · Difficulty {'▮'.repeat(frontmatter.difficulty) + '▯'.repeat(5 - frontmatter.difficulty)}</p>
    <h1 class="font-display text-5xl mt-2">{frontmatter.title}</h1>
    <div class="mt-3"><ComputeBadge tier={frontmatter.computeTier} /></div>
  </header>
  <article class="prose-book"><slot /></article>
</BookShell>
```

- [ ] **Step 2: Write `pages/weeks/[slug].astro`**

```astro
---
import { getCollection, type CollectionEntry } from 'astro:content'
import WeekLayout from '../../layouts/WeekLayout.astro'

export async function getStaticPaths() {
  const weeks = await getCollection('weeks')
  return weeks.map((entry) => ({ params: { slug: entry.data.slug }, props: { entry } }))
}
interface Props { entry: CollectionEntry<'weeks'> }
const { entry } = Astro.props
const { Content } = await entry.render()
---
<WeekLayout frontmatter={entry.data}>
  <Content />
</WeekLayout>
```

- [ ] **Step 3: Commit**

```bash
git add apps/book/src
git commit -m "feat(book): WeekLayout + dynamic /weeks/[slug] route"
```

---

## Task 14: Author Week 1 MDX (Linear Algebra) — the pipeline proof

**Files:**
- Create: `apps/book/src/content/weeks/week-01-linear-algebra.mdx`

**Interfaces:**
- Consumes: every content component from Tasks 8–11, MathBlock (Task 9), CodeBlock (Task 10), WeekLayout (Task 13).
- Produces: the first fully readable week + KaTeX SSR proof (unblocks Task 9's test) + reference-project handoff to Task 15.

- [ ] **Step 1: Author `week-01-linear-algebra.mdx`**

Write the full 8-part lesson. Below is the complete file — no placeholders. Copy verbatim.

```mdx
---
week: 1
part: 1
slug: "week-01-linear-algebra"
title: "Linear Algebra — the language of ML"
hook: "If you can describe your data as vectors and your models as matrices, deep learning stops being a black box."
hours: 20
computeTier: "green"
difficulty: 2
prereqSlugs: []
referenceProject: "projects/week-01-linalg-lab"
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
import CompanyPill from '../../components/callouts/CompanyPill.astro'

<Hook>Every model you will meet — every LLM, every image classifier, every recommender — reads its input as a list of numbers and its weights as a table of numbers. Learn the grammar of *lists* and *tables* and half of ML stops looking like magic.</Hook>

<Intuition>
A vector is a list of numbers with a *direction*. A matrix is a rule that takes a vector and returns another vector. That's the whole game.

<IntuitionCallout>
When GPT-4 embeds your prompt, it is turning your text into a **vector** in a 12,288-dimensional space. When it "thinks", it is repeatedly applying **matrices** to that vector.
</IntuitionCallout>

The four moves you must own by end of week: **dot product** (measures similarity), **matrix–vector product** (applies a rule), **eigendecomposition** (finds a matrix's axes), and **SVD** (breaks any matrix into three simpler pieces).
</Intuition>

## The math

The **dot product** between two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ is:

<MathBlock latex="\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \|\mathbf{a}\| \, \|\mathbf{b}\| \cos \theta" />

The right-hand form says: two vectors are similar when the angle between them is small. This is exactly how semantic search works — every embedding model is answering "which stored vectors have the largest dot product with my query vector?"

The **singular value decomposition** writes any real $m \times n$ matrix $\mathbf{A}$ as:

<MathBlock latex="\mathbf{A} = \mathbf{U}\,\mathbf{\Sigma}\,\mathbf{V}^\top" />

where $\mathbf{U}$ and $\mathbf{V}$ are orthogonal and $\mathbf{\Sigma}$ is diagonal with descending non-negative entries $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$. Truncating to the top $k$ singular values gives you the best rank-$k$ approximation to $\mathbf{A}$ — this is the same idea behind **LoRA**, PCA, and image compression.

<Gotcha>The order matters. $\mathbf{U}\mathbf{V}^\top \neq \mathbf{V}^\top \mathbf{U}$. If your gradients explode in Week 9, this is often why.</Gotcha>

## The code

Compute a dot product two ways — the slow explicit loop and the vectorized NumPy call — then time them.

```python
import numpy as np, time

def dot_slow(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

rng = np.random.default_rng(0)
a, b = rng.standard_normal(10_000), rng.standard_normal(10_000)

t = time.perf_counter(); dot_slow(a, b);      slow = time.perf_counter() - t
t = time.perf_counter(); np.dot(a, b);        fast = time.perf_counter() - t
print(f"loop: {slow*1000:.2f} ms, vectorized: {fast*1000:.4f} ms, speedup: {slow/fast:.0f}×")
```

Expect ~500× speedup. NumPy dispatches to BLAS, which is SIMD-vectorized C. Every framework you'll use — PyTorch, JAX — sits on the same idea.

<CompanyLens topic="How the frontier uses linear algebra">
The dot product is not a starter concept — it is the operation frontier labs spend billions optimizing.

- <CompanyPill company="OpenAI" tint="var(--violet-9)" /> Every attention head is a batched dot product. Their GPT training runs are, arithmetically, dot products in a trench coat.
- <CompanyPill company="Anthropic" tint="var(--amber-9)" /> Constitutional AI classifiers score model responses by dot product against a "helpfulness" vector.
- <CompanyPill company="Google DeepMind" tint="var(--indigo-9)" /> Gemini's TPUs are matrix-multiply engines — the entire chip is designed around one operation.
- <CompanyPill company="Meta AI" tint="var(--teal-9)" /> Llama's grouped-query attention shares key/value projections across heads to cut *how many* dot products are needed at inference.
- <CompanyPill company="xAI" tint="var(--slate-11)" /> Grok's Colossus cluster is 200k H100s executing dot products in parallel.
- <CompanyPill company="DeepSeek" tint="var(--tomato-9)" /> DeepSeek-V3's MLA (Multi-head Latent Attention) compresses the KV cache with a low-rank SVD-style decomposition.
</CompanyLens>

<ReferenceProject path="projects/week-01-linalg-lab" name="linalg-lab" hours="6–8 hrs">
The `linalg-lab` project contains two hands-on modules:

1. **Vector-space visualizer** — plot 2-D and 3-D vectors, watch dot-products, angles, and projections update as you drag.
2. **SVD image compressor** — load a photo, compute its SVD, watch the image reappear as you add rank-1 components one at a time. This is the exact idea behind LoRA.

Clone, run `uv sync`, then `pytest`.
</ReferenceProject>

<Assignments>
**Warm-up (30 min).** In the reference project, open `src/linalg_lab/vectors.py`, find `cosine_similarity`, and predict what happens if either input is the zero vector. Then run the test suite to see whether the code agrees with you.

**Build (2–3 hrs).** Extend `svd_compress.py` to accept any image file (currently only PNG). Add a CLI flag `--target-quality` that picks the smallest $k$ whose reconstruction MSE is below the target.

**Challenge (3+ hrs).** Implement a "least-squares logo remover": given an image with a watermark you want gone, ask the user to select a rectangular region, then reconstruct that region using a low-rank SVD approximation of the surrounding pixels. Portfolio-worthy if the output looks good.
</Assignments>

<InterviewDrill role="research-eng" company="Google DeepMind" time="20 min">
"Derive the closed-form solution to $\min_\mathbf{x} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$ and explain when the normal-equation form $\mathbf{x} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$ can be numerically dangerous."

<details>
<summary>Reveal worked solution</summary>

Take the gradient with respect to $\mathbf{x}$ of $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 = (\mathbf{A}\mathbf{x} - \mathbf{b})^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$:

<MathBlock latex="\nabla_\mathbf{x} = 2\mathbf{A}^\top(\mathbf{A}\mathbf{x} - \mathbf{b}) = 0 \implies \mathbf{A}^\top\mathbf{A}\,\mathbf{x} = \mathbf{A}^\top\mathbf{b}" />

If $\mathbf{A}^\top\mathbf{A}$ is invertible, $\mathbf{x} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$. **Danger:** $\text{cond}(\mathbf{A}^\top\mathbf{A}) = \text{cond}(\mathbf{A})^2$, so a mildly ill-conditioned $\mathbf{A}$ becomes catastrophically ill-conditioned. Prefer QR decomposition or SVD-based pseudoinverse in practice.
</details>
</InterviewDrill>

<FurtherReading>
- **Deep Learning**, Goodfellow, Bengio, Courville — Chapter 2 (Linear Algebra)
- **Introduction to Linear Algebra**, Gilbert Strang — 5th edition
- **The Matrix Cookbook** (Petersen & Pedersen, free PDF) — identities and derivatives
- **3Blue1Brown, *Essence of Linear Algebra*** (YouTube, ~3 hours) — the visual companion this week deserves
- **DeepSeek-V3 paper**, arXiv:2412.19437 — read §3.2 on Multi-head Latent Attention for a working example of low-rank decomposition in production
</FurtherReading>

<KeyTakeaways>
- Vectors are lists with direction; matrices are rules that map vectors to vectors.
- Dot product = cosine × magnitudes. Every semantic-similarity system reduces to this.
- SVD gives you the best low-rank approximation to any matrix — the mechanism behind LoRA and PCA.
- $\text{cond}(\mathbf{A}^\top\mathbf{A}) = \text{cond}(\mathbf{A})^2$ — never square a matrix casually.
- Frontier labs spend billions optimizing exactly the operations you learned this week.
</KeyTakeaways>
```

- [ ] **Step 2: Build the site and confirm the KaTeX SSR test now passes**

Flip the failing `test.fail()` marker in `apps/book/tests/katex-ssr.spec.ts` back to a passing `test(...)`, then:

```bash
pnpm --filter book test
```

Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add apps/book
git commit -m "feat(book): author Week 1 (Linear Algebra) — proves lesson pipeline"
```

---

## Task 15: Init the `linalg-lab` Python reference project

**Files:**
- Create: `projects/week-01-linalg-lab/pyproject.toml`
- Create: `projects/week-01-linalg-lab/src/linalg_lab/__init__.py`

**Interfaces:**
- Consumes: root uv workspace (Task 1).
- Produces: an importable Python package `linalg_lab` (empty for now) that later tasks fill in.

- [ ] **Step 1: Create the project directory**

```bash
mkdir -p projects/week-01-linalg-lab/src/linalg_lab projects/week-01-linalg-lab/tests projects/week-01-linalg-lab/notebooks projects/week-01-linalg-lab/assignments
```

- [ ] **Step 2: Write `projects/week-01-linalg-lab/pyproject.toml`**

```toml
[project]
name = "linalg-lab"
version = "0.1.0"
description = "Week 1 — vectors, matrices, SVD"
requires-python = ">=3.13"
dependencies = ["numpy>=2.1", "matplotlib>=3.9", "pillow>=10.4"]

[project.optional-dependencies]
dev = ["pytest>=8.3", "hypothesis>=6.112", "ruff>=0.7", "pyright>=1.1"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
addopts = "-v --strict-markers"
testpaths = ["tests"]
```

- [ ] **Step 3: Write `src/linalg_lab/__init__.py`**

```python
"""Week 1 — linear algebra sandbox."""
__version__ = "0.1.0"
```

- [ ] **Step 4: Sync the workspace**

```bash
uv sync
uv pip install -e "projects/week-01-linalg-lab[dev]"
```

Verify: `uv run python -c "import linalg_lab; print(linalg_lab.__version__)"` → `0.1.0`.

- [ ] **Step 5: Commit**

```bash
git add projects/week-01-linalg-lab pyproject.toml
git commit -m "chore(w01): scaffold linalg-lab python package"
```

---

## Task 16: Implement `vectors.py` (TDD)

**Files:**
- Create: `projects/week-01-linalg-lab/tests/test_vectors.py`
- Create: `projects/week-01-linalg-lab/src/linalg_lab/vectors.py`

**Interfaces:**
- Consumes: numpy.
- Produces: `dot(a, b)`, `cosine_similarity(a, b)`, `project(a, onto)` — all `ndarray → ndarray | float`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vectors.py`:

```python
import numpy as np
import pytest
from hypothesis import given, strategies as st
from linalg_lab.vectors import cosine_similarity, dot, project


def test_dot_of_orthogonal_is_zero():
    assert dot(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_dot_of_parallel_equals_product_of_norms():
    a = np.array([3.0, 4.0])
    assert dot(a, a) == pytest.approx(25.0)


def test_cosine_of_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_of_opposite_vectors_is_negative_one():
    v = np.array([1.0, 2.0])
    assert cosine_similarity(v, -v) == pytest.approx(-1.0)


def test_cosine_of_zero_vector_is_zero_by_convention():
    z = np.array([0.0, 0.0])
    assert cosine_similarity(z, np.array([1.0, 0.0])) == 0.0


def test_project_v_onto_x_axis_zeros_y():
    v = np.array([3.0, 4.0])
    x = np.array([1.0, 0.0])
    np.testing.assert_allclose(project(v, x), np.array([3.0, 0.0]))


@given(st.integers(min_value=1, max_value=100).flatmap(lambda n:
    st.tuples(st.lists(st.floats(-10, 10, allow_nan=False), min_size=n, max_size=n),
              st.lists(st.floats(-10, 10, allow_nan=False), min_size=n, max_size=n))))
def test_dot_is_commutative(vs):
    a, b = np.array(vs[0]), np.array(vs[1])
    assert dot(a, b) == pytest.approx(dot(b, a), abs=1e-9)
```

- [ ] **Step 2: Run and confirm they fail**

```bash
uv run pytest projects/week-01-linalg-lab/tests/test_vectors.py -v
```

Expected: ImportError (`vectors` doesn't exist yet).

- [ ] **Step 3: Implement `src/linalg_lab/vectors.py`**

```python
"""Vector operations from first principles."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dot(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Return the dot product of two vectors of equal length."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sum(a * b))


def cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Return cos(angle) in [-1, 1]. Zero vectors return 0 by convention."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def project(a: NDArray[np.floating], onto: NDArray[np.floating]) -> NDArray[np.floating]:
    """Project vector `a` onto vector `onto`."""
    denom = dot(onto, onto)
    if denom == 0.0:
        raise ValueError("cannot project onto zero vector")
    scalar = dot(a, onto) / denom
    return scalar * onto
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
uv run pytest projects/week-01-linalg-lab/tests/test_vectors.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Run ruff + pyright**

```bash
uv run ruff check projects/week-01-linalg-lab
uv run pyright projects/week-01-linalg-lab
```

Expected: no lint errors, no type errors.

- [ ] **Step 6: Commit**

```bash
git add projects/week-01-linalg-lab
git commit -m "feat(w01): vectors.py with property-based tests"
```

---

## Task 17: Implement `svd_compress.py` (TDD)

**Files:**
- Create: `projects/week-01-linalg-lab/tests/test_svd_compress.py`
- Create: `projects/week-01-linalg-lab/src/linalg_lab/svd_compress.py`
- Create: `projects/week-01-linalg-lab/notebooks/01-svd-images.ipynb`

**Interfaces:**
- Consumes: numpy, PIL.
- Produces: `svd_reconstruct(matrix, k)`, `compress_image(path, k)` — reconstructed matrix / grayscale image.

- [ ] **Step 1: Write failing tests**

Create `tests/test_svd_compress.py`:

```python
import numpy as np
from linalg_lab.svd_compress import svd_reconstruct


def test_rank_full_reconstruction_is_exact():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 4))
    np.testing.assert_allclose(svd_reconstruct(a, k=4), a, atol=1e-10)


def test_rank_1_reconstruction_has_rank_1():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((10, 8))
    approx = svd_reconstruct(a, k=1)
    # Numerical rank should be 1
    assert np.linalg.matrix_rank(approx, tol=1e-8) == 1


def test_error_decreases_monotonically_with_k():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((20, 15))
    errs = [np.linalg.norm(a - svd_reconstruct(a, k=k), 'fro') for k in range(1, 15)]
    for i in range(len(errs) - 1):
        assert errs[i] >= errs[i + 1] - 1e-9


def test_k_zero_returns_zeros():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(svd_reconstruct(a, k=0), np.zeros_like(a))
```

- [ ] **Step 2: Run and confirm they fail**

```bash
uv run pytest projects/week-01-linalg-lab/tests/test_svd_compress.py -v
```

- [ ] **Step 3: Implement `src/linalg_lab/svd_compress.py`**

```python
"""Rank-k SVD reconstruction — the arithmetic behind LoRA and PCA."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def svd_reconstruct(matrix: NDArray[np.floating], k: int) -> NDArray[np.floating]:
    """Return the best rank-k approximation of `matrix` in Frobenius norm."""
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return np.zeros_like(matrix, dtype=float)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    k_eff = min(k, s.shape[0])
    return (u[:, :k_eff] * s[:k_eff]) @ vt[:k_eff, :]


def compress_image(path: str | Path, k: int) -> NDArray[np.floating]:
    """Load a grayscale image and return its rank-k SVD reconstruction."""
    img = np.asarray(Image.open(path).convert("L"), dtype=float)
    return svd_reconstruct(img, k=k)
```

- [ ] **Step 4: Run tests, confirm they pass, run ruff+pyright**

```bash
uv run pytest projects/week-01-linalg-lab/tests -v
uv run ruff check projects/week-01-linalg-lab
uv run pyright projects/week-01-linalg-lab
```

Expected: 11 tests total PASS, no lint/type errors.

- [ ] **Step 5: Create the notebook `notebooks/01-svd-images.ipynb`**

A minimal Jupyter notebook with 4 cells:

```python
# cell 1
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
from linalg_lab.svd_compress import compress_image

# cell 2
sample_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
# For the notebook, expect the learner to `wget` this into ./data/ first.

# cell 3
for k in [1, 5, 20, 80]:
    plt.figure(); plt.title(f"k = {k}")
    plt.imshow(compress_image("data/cat.jpg", k=k), cmap="gray"); plt.axis("off")

# cell 4  — measure compression ratio
```

Save this as a `.ipynb` (JSON) file — use nbformat or hand-write it.

- [ ] **Step 6: Commit**

```bash
git add projects/week-01-linalg-lab
git commit -m "feat(w01): svd_compress.py with rank-k image reconstruction + notebook"
```

---

## Task 18: Author READMEs, notes, assignments — and OpenAI company profile

**Files:**
- Create: `projects/week-01-linalg-lab/README.md`, `SOLUTION_NOTES.md`, `COMPUTE.md`
- Create: `projects/week-01-linalg-lab/assignments/warmup.md`, `build.md`, `challenge.md`
- Create: `apps/book/src/content/companies/openai.mdx`

**Interfaces:**
- Consumes: nothing.
- Produces: complete W1 documentation + first company profile (needed for `/companies` route later, but the frontmatter alone unblocks any `CompanyPill` reference in W1 MDX).

- [ ] **Step 1: Write `projects/week-01-linalg-lab/README.md`**

```markdown
# linalg-lab — Week 1 reference project

**Goals:** Feel *at home* with vectors, matrices, dot products, projections, and SVD, in NumPy. Build the SVD image compressor the book chapter closes with.

## Prereqs
- Python 3.13
- uv (`brew install uv`)
- 200 MB free disk

## Run
```bash
uv sync
uv pip install -e ".[dev]"
uv run pytest
uv run jupyter lab notebooks/
```

## What's inside
- `src/linalg_lab/vectors.py` — dot, cosine similarity, projection.
- `src/linalg_lab/svd_compress.py` — SVD rank-k reconstruction, image compression.
- `tests/` — pytest + hypothesis property tests.
- `notebooks/01-svd-images.ipynb` — visual walkthrough.
- `assignments/{warmup,build,challenge}.md` — see the book chapter.
```

- [ ] **Step 2: Write `SOLUTION_NOTES.md`**

```markdown
# linalg-lab — design notes

## Why not `numpy.dot` everywhere?
We reimplement `dot` explicitly because the point of the assignment is to feel the sum. NumPy's `dot` calls BLAS SGEMV internally; the timing exercise in the chapter compares.

## Why SVD via `numpy.linalg.svd`?
`np.linalg.svd` calls LAPACK's `gesdd` — an industrial-strength divide-and-conquer routine. Writing a naive SVD by hand is a 3-week project. The chapter uses SVD as a *tool*; the arithmetic gets built later in W7 (unsupervised methods).

## Numerical gotchas
- Zero-length vector → cosine returns 0 by convention (avoid NaN).
- `np.linalg.svd(A, full_matrices=False)` — never pass `True` for tall matrices unless you want an $m \times m$ $\mathbf{U}$.
```

- [ ] **Step 3: Write `COMPUTE.md`**

```markdown
# Compute — Week 1

**Tier:** 🟢 Local (M-series or any laptop with Python 3.13).

Nothing here needs a GPU. NumPy hits Accelerate on macOS automatically.
```

- [ ] **Step 4: Write `assignments/warmup.md`, `build.md`, `challenge.md`**

Copy the three assignment blocks verbatim from `apps/book/src/content/weeks/week-01-linear-algebra.mdx` (§Assignments) into three files under `projects/week-01-linalg-lab/assignments/`. Match content exactly — the book chapter is the source of truth.

- [ ] **Step 5: Write `apps/book/src/content/companies/openai.mdx`**

```mdx
---
slug: "openai"
name: "OpenAI"
tint: "#7B61FF"
tagline: "RLHF pioneers, product-first"
founded: 2015
hq: "San Francisco"
---

## Philosophy

Product-first. Ship the model *as* the product; iterate in the open with users and use the learnings to feed the next training run. RLHF was the first bet that paid off; the o-series and Sora extend the same "scale + human feedback + iterated deployment" arc.

## Model timeline

- **2018** GPT-1 — 117M params, decoder-only transformer.
- **2019** GPT-2 — 1.5B params, "too dangerous to release."
- **2020** GPT-3 — 175B params, in-context learning.
- **2022** InstructGPT, ChatGPT — RLHF at scale.
- **2023** GPT-4, Whisper, DALL·E 3 — multimodal push.
- **2024** o1 — the first widely-deployed reasoning model, trained with RL on chain-of-thought.
- **2025-26** o3, Sora 2 — reasoning + generation converge.

## Loop guide (public information, ~2026)

5 rounds — recruiter screen · coding (ML-flavored, no LC hard mode) · ML debugging round · ML system design · behavioral + values. The debugging round is signature-OpenAI: they hand you a training loop that misbehaves and ask you to find the bug live.

## Reading list

- Every GPT tech report, in order.
- **InstructGPT** — arXiv:2203.02155 — the paper that made RLHF famous.
- **Whisper** — the audio companion; a good ASR reference.
- **GPT-4 System Card** — how to read a safety card.
- Any current OpenAI flagship system card (published quarterly).

## Interview angles seeded in this book

- W1 · The dot product is what attention *is*.
- W15b · o-series and test-time compute — the design philosophy.
- W19 · Prompt caching economics.
- W21 · Values screens — how to speak about deployment.
```

- [ ] **Step 6: Verify everything builds**

```bash
pnpm --filter book build
```

Expected: clean build, no MDX errors.

- [ ] **Step 7: Commit**

```bash
git add projects/week-01-linalg-lab apps/book/src/content/companies
git commit -m "docs(w01): READMEs, notes, assignments; add OpenAI company profile"
```

---

## Task 19: Vercel deployment config

**Files:**
- Create: `vercel.json`
- Modify: `apps/book/astro.config.mjs` (Vercel adapter)

**Interfaces:**
- Consumes: everything built so far.
- Produces: a green preview deploy at a Vercel URL.

- [ ] **Step 1: Add the Vercel adapter**

```bash
pnpm --filter book add @astrojs/vercel
```

- [ ] **Step 2: Update `astro.config.mjs`**

```js
import vercel from '@astrojs/vercel/static'

// inside defineConfig:
adapter: vercel(),
output: 'static'
```

- [ ] **Step 3: Write `vercel.json` (repo root)**

```json
{
  "version": 2,
  "buildCommand": "pnpm --filter book build",
  "installCommand": "pnpm install --frozen-lockfile",
  "outputDirectory": "apps/book/.vercel/output"
}
```

- [ ] **Step 4: Deploy (manual, one-time)**

```bash
npx vercel login   # interactive; the user runs this themselves
npx vercel --yes   # first-time link + preview deploy
```

Expected: preview URL like `https://core-ai-<hash>.vercel.app` prints and returns 200 for `/` and `/weeks/week-01-linear-algebra`.

- [ ] **Step 5: Commit**

```bash
git add vercel.json apps/book
git commit -m "chore: vercel adapter + deploy config"
```

---

## Task 20: CI + weekly link check

**Files:**
- Create: `.github/workflows/book.yml`, `projects.yml`, `ci.yml`, `link-check.yml`
- Create: `scripts/check-links.mjs`

**Interfaces:**
- Consumes: repo state.
- Produces: CI runs on every PR + weekly cron for link rot (Review Focus #4).

- [ ] **Step 1: Write `.github/workflows/book.yml`**

```yaml
name: book
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 24, cache: 'pnpm' }
      - uses: pnpm/action-setup@v4
        with: { version: 9 }
      - run: pnpm install --frozen-lockfile
      - run: pnpm --filter book build
      - uses: microsoft/playwright-github-action@v1
      - run: pnpm --filter book test
```

- [ ] **Step 2: Write `.github/workflows/projects.yml`**

```yaml
name: projects
on: [push, pull_request]
jobs:
  pytest:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest projects/
      - run: uv run ruff check projects/
      - run: uv run pyright projects/
```

- [ ] **Step 3: Write `.github/workflows/ci.yml` (root lint/format)**

```yaml
name: ci
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 24 }
      - uses: pnpm/action-setup@v4
        with: { version: 9 }
      - run: pnpm install --frozen-lockfile
      - run: pnpm lint
```

- [ ] **Step 4: Write `scripts/check-links.mjs`**

```js
#!/usr/bin/env node
import { readFile, readdir } from 'node:fs/promises'
import { join } from 'node:path'

async function* mdxFiles(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name)
    if (entry.isDirectory()) yield* mdxFiles(p)
    else if (p.endsWith('.mdx') || p.endsWith('.md')) yield p
  }
}

const urlRe = /https?:\/\/[^\s)>\]"']+/g
const seen = new Set()
const failed = []

for await (const f of mdxFiles('.')) {
  const text = await readFile(f, 'utf-8')
  for (const m of text.matchAll(urlRe)) {
    const url = m[0].replace(/[.,;:!)\]]+$/, '')
    if (seen.has(url)) continue
    seen.add(url)
    try {
      const r = await fetch(url, { method: 'HEAD', redirect: 'follow', signal: AbortSignal.timeout(8000) })
      if (!r.ok) failed.push({ url, status: r.status, file: f })
    } catch (e) {
      failed.push({ url, status: e.name, file: f })
    }
  }
}

if (failed.length) {
  console.error('BROKEN LINKS:')
  for (const x of failed) console.error(`  ${x.status}  ${x.url}   (in ${x.file})`)
  process.exit(1)
}
console.log(`OK — checked ${seen.size} unique URLs`)
```

- [ ] **Step 5: Write `.github/workflows/link-check.yml`**

```yaml
name: link-check
on:
  schedule: [{ cron: '0 8 * * 1' }]     # Monday 08:00 UTC
  workflow_dispatch:
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 24 }
      - run: node scripts/check-links.mjs
```

- [ ] **Step 6: Locally, dry-run the link checker**

```bash
node scripts/check-links.mjs
```

Expected: exit 0 (assuming the arXiv/wikipedia URLs in W1 all resolve).

- [ ] **Step 7: Commit**

```bash
git add .github scripts
git commit -m "ci: book/projects/ci workflows + weekly link-check cron"
```

---

## Task 21: Final QA + production deploy

**Files:** none new.

**Interfaces:** consumes everything.

- [ ] **Step 1: Full clean build + test locally**

```bash
pnpm install --frozen-lockfile
pnpm build
pnpm test
uv run pytest projects/
```

Every step exits 0.

- [ ] **Step 2: Manual QA checklist (run at http://localhost:4321 after `pnpm --filter book preview`)**

Confirm each item, then check the box:

- [ ] Home page renders hero + Month 1 with W1 in the roadmap
- [ ] "Start Week 1 →" links to `/weeks/week-01-linear-algebra`
- [ ] Week 1 renders **all 9 sections** (Hook, Intuition, Math, Code, CompanyLens, ReferenceProject, Assignments, InterviewDrill, FurtherReading, KeyTakeaways)
- [ ] Math renders via KaTeX (view source: `<span class="katex">` present)
- [ ] Code blocks have syntax colors, file affordances, and a copy button
- [ ] The compute-tier badge (🟢) shows at the top of Week 1
- [ ] Dark-mode toggle works (localStorage persists between reloads)
- [ ] Mobile (375px, DevTools) has zero horizontal scroll on all rendered routes
- [ ] `/companies/openai` — deferred; not routed yet (Plan 2). Note this as expected.
- [ ] `projects/week-01-linalg-lab/` — clone into a fresh dir, `uv sync && uv run pytest` — 11 tests pass

- [ ] **Step 3: Promote to production**

```bash
npx vercel --prod
```

- [ ] **Step 4: Tag the milestone**

```bash
git tag -a v0.1.0-week01 -m "Plan 1 complete: platform + W1 pipeline proof"
git push --tags
```

- [ ] **Step 5: Final commit (empty, marker)**

```bash
git commit --allow-empty -m "chore: plan 1 complete — foundation + Week 1 shipped"
```

---

## What Plan 1 leaves for Plan 2 (foreshadowed, not built)

- `/companies/[slug]` dynamic route + all 7 profiles (Anthropic, DeepMind, Meta, xAI, DeepSeek, Qwen — plus finishing OpenAI's blog references).
- `/companies` matrix landing + `/companies/compare` shell.
- `/interview` dashboard route (Preact + Signals island with localStorage progress).
- MicroRecall + WeeklyQuiz Preact islands.
- Motion v12 scroll-linked SVG diagrams (concept animations).
- Full CompanyLens 5-block template with **all 7 companies** in every week's Lens.
- Pagefind client-side search.
- `packages/viz/` — shared SVG primitives (VectorPlayground, MatrixMul, AttentionHeatmap).
- Companion routes: `/how-to-study`, `/glossary`, `/math-primer`, `/paper-reading-protocol`, `/tech-writing`.

## What Plan 3+ ships

- Plan 3: **Weeks 2–4** (calculus, probability, python+info theory) + their 3 reference projects.
- Plan 4: **Weeks 5–8** (classical ML).
- Plan 5: **Weeks 9–12** (deep learning).
- Plan 6: **Weeks 13–17** (transformers/LLMs) — includes W15a/W15b/W17 with cloud-runbook docs.
- Plan 7: **Weeks 18–21** (systems + safety).
- Plan 8: **Weeks 22–25** (specialization + capstone kits + interview prep polish).
