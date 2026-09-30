import { expect, test } from '@playwright/test'

const WEEKS = [
  'week-01-linear-algebra',
  'week-02-calculus',
  'week-03-probability',
  'week-04-python-info-theory',
  'week-05-linreg-from-scratch',
  'week-06-xgboost-kaggle',
  'week-07-unsupervised-viz',
  'week-08-ml-eval-suite',
  'week-09-mini-torch',
  'week-10-cifar-resnet',
  'week-11-char-rnn-attention',
  'week-12-rl-gridworld',
  'week-13-nano-gpt-ssm',
  'week-14-mini-bpe-pretrain',
  'week-15a-sft-lora-dpo-lab',
  'week-15b-moe-and-reasoning',
  'week-17-mini-rag-multimodal',
]
const ORDER = ['OpenAI', 'Anthropic', 'Google DeepMind', 'Meta AI (FAIR)', 'xAI', 'DeepSeek', 'Alibaba Qwen']

for (const slug of WEEKS) {
  test(`${slug} CompanyLens renders all 7 companies in fixed order`, async ({ page }) => {
    await page.goto(`/weeks/${slug}`)
    const rows = await page
      .locator('section:has(h2:has-text("Company Lens")) tbody tr')
      .allTextContents()
    expect(rows).toHaveLength(7)
    ORDER.forEach((name, i) => expect(rows[i]).toContain(name))
  })
}
