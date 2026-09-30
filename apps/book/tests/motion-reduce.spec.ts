import { expect, test } from '@playwright/test'

test('reduced-motion content is visible with no transform after hydration', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/motion-probe', { waitUntil: 'networkidle' })
  const box = page.getByTestId('reveal-box')
  await expect(box).toBeVisible()
  await expect.poll(() => box.evaluate((el) => getComputedStyle(el).opacity)).toBe('1')
  await expect.poll(() => box.evaluate((el) => getComputedStyle(el).transform)).toBe('none')
})

test('below-fold reveal becomes visible on intersection and preference changes', async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: 'no-preference' })
  await page.goto('/motion-probe?belowFold=1', { waitUntil: 'networkidle' })
  const box = page.getByTestId('reveal-box')
  await expect
    .poll(() => box.evaluate((el) => Number(getComputedStyle(el).opacity)))
    .toBeLessThan(1)
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await expect.poll(() => box.evaluate((el) => getComputedStyle(el).opacity)).toBe('1')
  await expect.poll(() => box.evaluate((el) => getComputedStyle(el).transform)).toBe('none')
})

test('server-rendered reveal content remains readable without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false })
  const page = await context.newPage()
  await page.goto('http://localhost:4321/motion-probe')
  const box = page.getByTestId('reveal-box')
  await expect(box).toBeVisible()
  expect(await box.evaluate((el) => getComputedStyle(el).opacity)).toBe('1')
  await context.close()
})
