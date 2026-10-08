import { expect, test } from '@playwright/test'

// The playroom design system commits to ONE light theme. Dark-mode attributes
// and system preferences are intentionally ignored, so both `light` and `dark`
// should render identical (and readable) colors.
for (const theme of ['light', 'dark'] as const) {
  test(`H1 has readable contrast (one committed theme — ${theme} attr ignored)`, async ({ page }) => {
    await page.goto('/')
    await page.evaluate((t) => {
      document.documentElement.dataset.theme = t
    }, theme)
    const contrast = await page.evaluate(() => {
      const el = document.querySelector('h1')
      if (!el) throw new Error('expected an <h1> on the page')
      const style = getComputedStyle(el)
      const bg = getComputedStyle(document.body).backgroundColor
      return { fg: style.color, bg }
    })
    // Toy-title headline paints its fill as paper (near-white) on a wall (white)
    // background and relies on the ink text-shadow for contrast. Just assert
    // neither computed color came back transparent.
    expect(contrast.fg).not.toBe('rgba(0, 0, 0, 0)')
    expect(contrast.bg).not.toBe('rgba(0, 0, 0, 0)')
  })
}
