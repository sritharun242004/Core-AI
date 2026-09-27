export type CompanySlug =
  | 'openai'
  | 'anthropic'
  | 'deepmind'
  | 'meta'
  | 'xai'
  | 'deepseek'
  | 'qwen'

export interface Company {
  slug: CompanySlug
  name: string
  tint: string
  emoji: string
}

// Fixed order per spec §4.1 — never re-sort.
export const COMPANIES: readonly Company[] = [
  { slug: 'openai',    name: 'OpenAI',           tint: '#7B61FF', emoji: '🟣' },
  { slug: 'anthropic', name: 'Anthropic',        tint: '#D97757', emoji: '🟠' },
  { slug: 'deepmind',  name: 'Google DeepMind',  tint: '#4285F4', emoji: '🔵' },
  { slug: 'meta',      name: 'Meta AI (FAIR)',   tint: '#1877F2', emoji: '🟢' },
  { slug: 'xai',       name: 'xAI',              tint: '#1E1E1E', emoji: '⚫' },
  { slug: 'deepseek',  name: 'DeepSeek',         tint: '#E4B04A', emoji: '🟡' },
  { slug: 'qwen',      name: 'Alibaba Qwen',     tint: '#FF6A00', emoji: '🟠' },
] as const

export const COMPANY_SLUGS = COMPANIES.map((c) => c.slug) as readonly CompanySlug[]

export function assertAll7<T>(
  record: Partial<Record<CompanySlug, T>>,
  block: string,
): asserts record is Record<CompanySlug, T> {
  const missing = COMPANY_SLUGS.filter((s) => !(s in record))
  if (missing.length > 0) {
    throw new Error(`CompanyLens ${block}: missing entries for [${missing.join(', ')}]`)
  }
}
