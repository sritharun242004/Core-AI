import { expect, test } from '@playwright/test'
import { readFile, readdir } from 'node:fs/promises'
import { join } from 'node:path'

// Each pattern asserts a WRONG affirmative attribution. Disclaimers of the form
// "not Meta" / "NOT DeepMind" / "not from OpenAI" are excluded via a tempered
// greedy sub-pattern before the wrong-target.
const FORBIDDEN_STRINGS: [string, RegExp][] = [
  ['DPO must be attributed to Stanford, not Meta',
    /\bDPO\b(?:(?!\bnot\s+Meta\b).){0,60}\bMeta\b/i],
  ['A2A/ADK must be attributed to Google Cloud, not DeepMind',
    /\b(?:A2A|ADK)\b(?:(?!\bnot\s+DeepMind\b).){0,80}\bDeepMind\b/i],
  ['No Colossus peer-reviewed paper exists',
    /Colossus\s+paper|Colossus\s+arXiv/i],
  ['Chinchilla is DeepMind (not OpenAI/Anthropic)',
    /Chinchilla(?:(?!\bnot\s+(?:OpenAI|Anthropic)\b).){0,60}\b(?:OpenAI|Anthropic)\b/i],
  ['Age of AI has 3 authors (Kissinger, Schmidt, Huttenlocher)',
    /Age of AI[^.]{0,80}Kissinger[^.]{0,80}Schmidt(?!.{0,80}Huttenlocher)/i],
]

test('no company MDX file violates the attribution rules', async () => {
  const dir = join(process.cwd(), 'src/content/companies')
  const files = (await readdir(dir)).filter((f) => f.endsWith('.mdx'))
  expect(files.length).toBe(7)
  for (const f of files) {
    const body = await readFile(join(dir, f), 'utf-8')
    for (const [reason, pattern] of FORBIDDEN_STRINGS) {
      const m = body.match(pattern)
      if (m) {
        throw new Error(`${f}: ${reason} — matched /${pattern.source}/ → "${m[0]}"`)
      }
    }
  }
})
