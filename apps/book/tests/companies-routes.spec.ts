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
