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
  'week-18-fsdp-ring-lab',
  'week-19-vllm-benchmark',
  'week-20-evals-mlops-pipeline',
  'week-21-alignment-lab',
  'week-22l-agents-lab',
  'week-23l-mcp-a2a-adk-lab',
  'week-22p-two-tower-recsys',
  'week-23p-learning-to-rank-timeseries',
  'week-22r-transformer-repro',
  'week-23r-scaling-dpo-repro',
  'week-24-capstone',
  'week-25-interview-prep',
]
const ORDER = [
  'OpenAI',
  'Anthropic',
  'Google DeepMind',
  'Meta AI (FAIR)',
  'xAI',
  'DeepSeek',
  'Alibaba Qwen',
]

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
