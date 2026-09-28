import { expect, test } from '@playwright/test'

const CASES = [
  { slug: 'week-05-linreg-from-scratch', title: 'Linear and logistic regression' },
  { slug: 'week-06-xgboost-kaggle', title: 'Trees, ensembles, and XGBoost' },
  { slug: 'week-07-unsupervised-viz', title: 'Unsupervised learning' },
  { slug: 'week-08-ml-eval-suite', title: 'Evaluation' },
]

for (const c of CASES) {
  test(`/weeks/${c.slug} renders the classical-ML lesson anatomy`, async ({ page }) => {
    await page.goto(`/weeks/${c.slug}`)
    await expect(page.locator('main h1')).toContainText(c.title)
    await expect(page.getByRole('heading', { name: /Reference Project/i })).toBeVisible()
    await expect(page.getByRole('heading', { name: /^Assignments$/i })).toBeVisible()
    await expect(page.getByRole('heading', { name: /Company Lens/i })).toBeVisible()
  })
}
