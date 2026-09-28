import { expect, test } from '@playwright/test'

const WEEKS = [
  'week-01-linear-algebra',
  'week-02-calculus',
  'week-03-probability',
  'week-04-python-info-theory',
]
const ORDER = ['OpenAI', 'Anthropic', 'Google DeepMind', 'Meta AI (FAIR)', 'xAI', 'DeepSeek', 'Alibaba Qwen']

for (const slug of WEEKS) {
  test(`${slug} CompanyLens renders all 7 companies in fixed order`, async ({ page }) => {
    await page.goto(`/weeks/${slug}`)
    const rows = await page
      .locator('section:has(h2:has-text("Company Lens")) tbody tr')
      .allTextContents()
    expect(rows).toHaveLength(7)
    ORDER.forEach((name, i) => expect(rows[i]).toContain(name))
  })
}
