import { expect, test } from '@playwright/test'

test('/interview renders the dashboard with zero progress', async ({ page }) => {
  await page.goto('/interview')
  await expect(page.getByRole('heading', { name: 'Interview' })).toBeVisible()
  await expect(page.getByText('Total problems solved')).toBeVisible()
  await expect(page.getByText('Answer some quiz questions first')).toBeVisible()
})

test('/interview reflects a seeded localStorage progress record', async ({ page }) => {
  await page.goto('/interview')
  await page.evaluate(() => {
    localStorage.setItem(
      'core-ai:progress',
      JSON.stringify({
        answers: {
          'week1-q0': { attempts: 2, correct: 2 },
          'coding:lru': { attempts: 1, correct: 1 },
          'sysdes:recsys': { attempts: 1, correct: 1 },
        },
      }),
    )
  })
  await page.reload()
  await expect(page.locator('p.font-display.text-4xl')).toContainText('3')
})

test('/interview/coding-set has all 20 problems', async ({ page }) => {
  await page.goto('/interview/coding-set')
  const items = page.locator('main ol li')
  await expect(items).toHaveCount(20)
})
