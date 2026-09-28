import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { expect, test } from '@playwright/test'

const SLUGS = [
  'week-09-mini-torch',
  'week-10-cifar-resnet',
  'week-11-char-rnn-attention',
  'week-12-rl-gridworld',
]

for (const slug of SLUGS) {
  test(`KaTeX renders at build time on /${slug}`, async () => {
    const html = await readFile(
      join(process.cwd(), 'dist', 'client', 'weeks', slug, 'index.html'),
      'utf-8',
    )
    expect(html).toMatch(/<span class="katex/)
  })
}
