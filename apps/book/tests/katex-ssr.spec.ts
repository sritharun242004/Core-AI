import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

test('KaTeX renders at build time (SSR, no client JS)', async () => {
  const html = await readFile(join(process.cwd(), 'dist', 'weeks', 'week-01-linear-algebra', 'index.html'), 'utf-8')
  expect(html).toMatch(/<span class="katex/)
})
