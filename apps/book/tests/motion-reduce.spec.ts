import { expect, test } from '@playwright/test'

test.describe('ScrollReveal respects prefers-reduced-motion', () => {
  test('under reduce, children are visible immediately with no transform', async ({ page, context }) => {
    await context.emulateMedia({ reducedMotion: 'reduce' })
    await page.goto('/motion-probe')
    const box = page.getByTestId('reveal-box')
    await expect(box).toBeVisible()
    const transform = await box.evaluate((el) => getComputedStyle(el).transform)
    // no motion means either 'none' or the identity matrix
    expect(['none', 'matrix(1, 0, 0, 1, 0, 0)']).toContain(transform)
    const opacity = await box.evaluate((el) => Number(getComputedStyle(el).opacity))
    expect(opacity).toBe(1)
  })

  test('without reduce, initial opacity is < 1 before viewport intersect', async ({ page, context }) => {
    await context.emulateMedia({ reducedMotion: 'no-preference' })
    await page.goto('/motion-probe?belowFold=1')
    const box = page.getByTestId('reveal-box')
    // scroll to top so the reveal box is below the viewport
    await page.evaluate(() => window.scrollTo(0, 0))
    const opacity = await box.evaluate((el) => Number(getComputedStyle(el).opacity))
    expect(opacity).toBeLessThan(1)
  })
})
