import { expect, test } from '@playwright/test'

test('search opens, returns built-index results, and closes with Escape', async ({ page }) => {
  await page.goto('/search')
  const dialog = page.getByRole('dialog', { name: 'Search the book' })
  await expect(dialog).toBeVisible()
  await dialog.getByRole('searchbox', { name: 'Search query' }).fill('linear algebra')
  await expect(dialog.getByRole('link').first()).toBeVisible({ timeout: 15000 })
  await page.keyboard.press('Escape')
  await expect(dialog).not.toBeVisible()
})

test('missing search index shows a recoverable error, not zero matches', async ({ page }) => {
  await page.route('**/pagefind/**', (route) => route.abort())
  await page.goto('/search')
  const dialog = page.getByRole('dialog', { name: 'Search the book' })
  await expect(dialog).toBeVisible()
  await dialog.getByRole('searchbox', { name: 'Search query' }).fill('attention')
  await expect(dialog.getByText(/Search is unavailable/)).toBeVisible()
  await expect(dialog.getByRole('button', { name: 'Retry search' })).toBeVisible()
})
