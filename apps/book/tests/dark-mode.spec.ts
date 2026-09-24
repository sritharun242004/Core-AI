import { expect, test } from '@playwright/test'

for (const theme of ['light', 'dark'] as const) {
  test(`H1 has readable contrast in ${theme} theme`, async ({ page }) => {
    await page.goto('/')
    await page.evaluate((t) => {
      document.documentElement.dataset.theme = t
    }, theme)
    const contrast = await page.evaluate(() => {
      const el = document.querySelector('h1')
      if (!el) throw new Error('expected an <h1> on the page')
      const style = getComputedStyle(el)
      const bg = getComputedStyle(document.body).backgroundColor
      // crude parse; just assert neither is transparent and colors differ
      return { fg: style.color, bg }
    })
    expect(contrast.fg).not.toBe(contrast.bg)
    expect(contrast.fg).not.toBe('rgba(0, 0, 0, 0)')
    expect(contrast.bg).not.toBe('rgba(0, 0, 0, 0)')
  })
}
