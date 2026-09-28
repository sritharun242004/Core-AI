import { expect, test } from '@playwright/test'

const CASES = [
  { slug: 'week-02-calculus',           title: 'Calculus' },
  { slug: 'week-03-probability',        title: 'Probability & Statistics' },
  { slug: 'week-04-python-info-theory', title: 'Python for ML + Information Theory' },
]

for (const c of CASES) {
  test(`/weeks/${c.slug} renders + shows reference project + assignments`, async ({ page }) => {
    await page.goto(`/weeks/${c.slug}`)
    await expect(page.locator('main h1')).toContainText(c.title)
    await expect(page.getByRole('heading', { name: /Reference Project/i })).toBeVisible()
    await expect(page.getByRole('heading', { name: /^Assignments$/i })).toBeVisible()
  })
}
