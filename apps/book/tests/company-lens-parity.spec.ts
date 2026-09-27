import { expect, test } from '@playwright/test'

test('week 1 CompanyLens renders all 7 companies in fixed order', async ({ page }) => {
  await page.goto('/weeks/week-01-linear-algebra')
  const rows = await page.locator('section:has(h2:has-text("Company Lens")) tbody tr').allTextContents()
  const order = ['OpenAI', 'Anthropic', 'Google DeepMind', 'Meta AI (FAIR)', 'xAI', 'DeepSeek', 'Alibaba Qwen']
  expect(rows).toHaveLength(7)
  order.forEach((name, i) => expect(rows[i]).toContain(name))
})
