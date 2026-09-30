export interface WeekNavigationEntry {
  week: number
  slug: string
}

export interface WeekColumn extends WeekNavigationEntry {
  label: string
}

const splitWeeks: WeekColumn[] = [
  { week: 15, slug: 'week-15a-sft-lora-dpo-lab', label: '15a' },
  { week: 15, slug: 'week-15b-moe-and-reasoning', label: '15b' },
]

export const WEEK_COLUMNS: WeekColumn[] = [
  ...Array.from({ length: 14 }, (_, index) => {
    const week = index + 1
    return {
      week,
      slug: `week-${String(week).padStart(2, '0')}-${
        [
          'linear-algebra',
          'calculus',
          'probability',
          'python-info-theory',
          'linreg-from-scratch',
          'xgboost-kaggle',
          'unsupervised-viz',
          'ml-eval-suite',
          'mini-torch',
          'cifar-resnet',
          'char-rnn-attention',
          'rl-gridworld',
          'nano-gpt-ssm',
          'mini-bpe-pretrain',
        ][index]
      }`,
      label: String(week).padStart(2, '0'),
    }
  }),
  ...splitWeeks,
  ...Array.from({ length: 9 }, (_, index) => {
    const week = index + 17
    return { week, slug: `week-${String(week).padStart(2, '0')}-pending`, label: String(week) }
  }),
]

export function weekLabel(entry: WeekNavigationEntry): string {
  const match = entry.slug.match(/^week-(\d{2})([ab])?-/)
  if (!match) return String(entry.week).padStart(2, '0')
  return `${match[1]}${match[2] ?? ''}`
}

export function compareWeeks(a: WeekNavigationEntry, b: WeekNavigationEntry): number {
  if (a.week !== b.week) return a.week - b.week
  return weekLabel(a).localeCompare(weekLabel(b))
}
