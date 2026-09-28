import { expect, test } from '@playwright/test'

const CASES = [
  { slug: 'week-09-mini-torch', title: 'Neural nets from scratch' },
  { slug: 'week-10-cifar-resnet', title: 'CNNs and ResNets' },
  { slug: 'week-11-char-rnn-attention', title: 'RNNs, LSTMs, and attention' },
  { slug: 'week-12-rl-gridworld', title: 'Optimizers and reinforcement learning' },
]

for (const c of CASES) {
  test(`/weeks/${c.slug} renders the deep-learning lesson anatomy`, async ({ page }) => {
    await page.goto(`/weeks/${c.slug}`)
    await expect(page.locator('main h1')).toContainText(c.title)
    await expect(page.getByRole('heading', { name: /Reference Project/i })).toBeVisible()
    await expect(page.getByRole('heading', { name: /^Assignments$/i })).toBeVisible()
    await expect(page.getByRole('heading', { name: /Company Lens/i })).toBeVisible()
  })
}
