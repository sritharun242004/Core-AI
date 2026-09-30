import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { expect, test } from '@playwright/test'

const SLUGS = [
  'week-18-fsdp-ring-lab',
  'week-19-vllm-benchmark',
  'week-20-evals-mlops-pipeline',
  'week-21-alignment-lab',
]

for (const slug of SLUGS) {
  test(`${slug} renders the complete systems lesson and math`, async ({ page }) => {
    await page.goto(`/weeks/${slug}`)
    await expect(page.locator('main h1')).toBeVisible()
    for (const title of [/Reference Project/, /^Assignments$/, /Company Lens/, /Interview Drill/]) {
      await expect(page.locator('main h2.font-display').filter({ hasText: title })).toBeVisible()
    }
    const html = await readFile(
      join(process.cwd(), 'dist', 'client', 'weeks', slug, 'index.html'),
      'utf-8',
    )
    expect(html).toContain('class="katex')
    expect(html).not.toContain('katex-error')
  })
}

test('systems diagrams are accessible local assets', async ({ page }) => {
  await page.goto('/weeks/week-18-fsdp-ring-lab')
  const diagram = page.locator('main img[src="/diagrams/week-18-fsdp-ring.svg"]')
  await expect(diagram).toHaveAttribute('alt', /FSDP/)
  const response = await page.request.get('/diagrams/week-18-fsdp-ring.svg')
  expect(response.ok()).toBeTruthy()
})
