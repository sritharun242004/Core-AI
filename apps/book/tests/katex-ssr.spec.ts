import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

// Skipped: no MDX with math exists yet (week-01-linear-algebra.mdx lands in Task 14).
// Task 14 removes this .skip once the MDX is authored so the build actually produces
// dist/weeks/week-01-linear-algebra/index.html for this assertion to read.
test.skip('KaTeX renders at build time (SSR, no client JS)', async () => {
  const html = await readFile(join(process.cwd(), 'dist', 'weeks', 'week-01-linear-algebra', 'index.html'), 'utf-8')
  expect(html).toMatch(/<span class="katex/)
})
