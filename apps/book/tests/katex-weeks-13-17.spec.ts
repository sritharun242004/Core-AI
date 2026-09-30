import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { expect, test } from '@playwright/test'

const SLUGS = [
  'week-13-nano-gpt-ssm',
  'week-14-mini-bpe-pretrain',
  'week-15a-sft-lora-dpo-lab',
  'week-15b-moe-and-reasoning',
  'week-17-mini-rag-multimodal',
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
