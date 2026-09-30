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
    const suffixes = [
      'mini-rag-multimodal',
      'fsdp-ring-lab',
      'vllm-benchmark',
      'evals-mlops-pipeline',
      'alignment-lab',
      'tracks',
      'tracks',
      'capstone',
      'interview-prep',
    ]
    return { week, slug: `week-${week}-${suffixes[index]}`, label: String(week) }
  }),
]

export function weekLabel(entry: WeekNavigationEntry): string {
  const match = entry.slug.match(/^week-(\d{2})([ablpr])?-/)
  if (!match) return String(entry.week).padStart(2, '0')
  const suffix = match[2] ?? ''
  return `${match[1]}${entry.week >= 22 ? suffix.toUpperCase() : suffix}`
}

export function compareWeeks(a: WeekNavigationEntry, b: WeekNavigationEntry): number {
  if (a.week !== b.week) return a.week - b.week
  return weekLabel(a).localeCompare(weekLabel(b)) || a.slug.localeCompare(b.slug)
}

/** A roadmap column can contain three track choices, never a last-write-wins map. */
export function lessonsForColumn<T extends WeekNavigationEntry>(
  column: WeekColumn,
  entries: T[],
): T[] {
  return entries
    .filter(
      (entry) => entry.week === column.week && (entry.week !== 15 || entry.slug === column.slug),
    )
    .sort(compareWeeks)
}
