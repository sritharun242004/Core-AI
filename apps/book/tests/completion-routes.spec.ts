import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { expect, test } from '@playwright/test'

const SLUGS = [
  'week-22l-agents-lab',
  'week-23l-mcp-a2a-adk-lab',
  'week-22p-two-tower-recsys',
  'week-23p-learning-to-rank-timeseries',
  'week-22r-transformer-repro',
  'week-23r-scaling-dpo-repro',
  'week-24-capstone',
  'week-25-interview-prep',
]

for (const slug of SLUGS) {
  test(`${slug} has complete lesson anatomy and a real repository link`, async ({ page }) => {
    await page.goto(`/weeks/${slug}`)
    await expect(page.locator('main h1')).toBeVisible()
    for (const title of [/Reference Project/, /^Assignments$/, /Company Lens/, /Interview Drill/]) {
      await expect(page.locator('main h2.font-display').filter({ hasText: title })).toBeVisible()
    }
    await expect(
      page.locator(
        'main a[href^="https://github.com/sritharun242004/Core-AI/tree/plan-1-foundation/"]',
      ),
    ).toBeVisible()
    const html = await readFile(
      join(process.cwd(), 'dist/client/weeks', slug, 'index.html'),
      'utf8',
    )
    expect(html).toContain('class="katex')
    expect(html).not.toContain('katex-error')
  })
}

test('matrix preserves all six track links in 25 roadmap columns', async ({ page }) => {
  await page.goto('/companies')
  const table = page.locator('main table')
  await expect(table.locator('thead th')).toHaveCount(26)
  for (const slug of SLUGS.slice(0, 6)) {
    await expect(table.locator(`a[href="/weeks/${slug}"]`)).toHaveCount(1)
  }
  await expect(table.locator('tbody td')).toHaveCount(175)
  expect(await table.locator('tbody td').allTextContents()).not.toContain('·')
})

test('track selection and final lessons fit a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 })
  for (const path of ['/', '/weeks/week-24-capstone', '/weeks/week-25-interview-prep']) {
    await page.goto(path)
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
  }
})
