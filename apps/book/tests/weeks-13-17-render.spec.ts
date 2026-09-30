import { expect, test } from '@playwright/test'

const CASES = [
  { slug: 'week-13-nano-gpt-ssm', title: 'Causal attention and state-space models' },
  { slug: 'week-14-mini-bpe-pretrain', title: 'Tokenization and pretraining' },
  { slug: 'week-15a-sft-lora-dpo-lab', title: 'Week 15a' },
  { slug: 'week-15b-moe-and-reasoning', title: 'Week 15b' },
  { slug: 'week-17-mini-rag-multimodal', title: 'Multimodal systems and retrieval' },
]

for (const c of CASES) {
  test(`/weeks/${c.slug} renders the transformer/LLM lesson anatomy`, async ({ page }) => {
    await page.goto(`/weeks/${c.slug}`)
    await expect(page.locator('main h1')).toContainText(c.title)
    await expect(
      page.locator('main h2.font-display').filter({ hasText: /Reference Project/ }),
    ).toBeVisible()
    await expect(
      page.locator('main h2.font-display').filter({ hasText: /^Assignments$/ }),
    ).toBeVisible()
    await expect(
      page.locator('main h2.font-display').filter({ hasText: /Company Lens/ }),
    ).toBeVisible()
  })
}
