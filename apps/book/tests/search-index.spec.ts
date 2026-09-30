import { readFile, readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { expect, test } from '@playwright/test'

test('pagefind index exists and every expected route has an emitted HTML page', async () => {
  const clientDir = join(process.cwd(), 'dist', 'client')
  const pagefindDir = join(clientDir, 'pagefind')
  const files = await readdir(pagefindDir)
  expect(files).toContain('pagefind.js')

  const mustHave = [
    'index.html',
    'weeks/week-01-linear-algebra/index.html',
    'companies/index.html',
    'companies/openai/index.html',
    'companies/anthropic/index.html',
    'companies/deepmind/index.html',
    'companies/meta/index.html',
    'companies/xai/index.html',
    'companies/deepseek/index.html',
    'companies/qwen/index.html',
    'interview/index.html',
    'interview/coding-set/index.html',
    'how-to-study/index.html',
    'glossary/index.html',
    'math-primer/index.html',
    'paper-reading-protocol/index.html',
    'tech-writing/index.html',
  ]
  for (const rel of mustHave) {
    const html = await readFile(join(clientDir, rel), 'utf-8').catch(() => null)
    expect.soft(html, `expected dist/client/${rel} to exist for pagefind to index`).not.toBeNull()
  }
})
