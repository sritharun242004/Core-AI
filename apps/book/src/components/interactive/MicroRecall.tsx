import { signal } from '@preact/signals'
import { recordAnswer } from '../../lib/progress'

export interface RecallQuestion {
  id: string
  prompt: string
  choices: string[]
  correct: number
  explain: string
}

export interface MicroRecallProps { questions: RecallQuestion[] }

export function MicroRecall({ questions }: MicroRecallProps) {
  const idx = signal(0)
  const pick = signal<number | null>(null)

  const q = () => questions[idx.value]
  const onPick = (i: number) => {
    pick.value = i
    recordAnswer(q().id, i === q().correct)
  }
  const next = () => {
    idx.value = (idx.value + 1) % questions.length
    pick.value = null
  }

  return (
    <aside class="my-6 rounded-md border-l-4 border-accent-p1 bg-canvas-subtle p-4">
      <div class="flex items-start gap-3">
        <span aria-hidden="true" class="text-xl">💡</span>
        <div style={{ flex: 1 }}>
          <p class="font-medium">{q().prompt}</p>
          <ul class="mt-3 space-y-2">
            {q().choices.map((c, i) => {
              const chosen = pick.value === i
              const revealed = pick.value !== null
              const isRight = i === q().correct
              const cls =
                !revealed  ? 'border-border-soft' :
                isRight    ? 'border-accent-p5 bg-accent-p5/10' :
                chosen     ? 'border-accent-p6 bg-accent-p6/10' :
                             'border-border-soft opacity-50'
              return (
                <li>
                  <button
                    type="button"
                    disabled={revealed}
                    onClick={() => onPick(i)}
                    class={`w-full text-left px-3 py-2 rounded-sm border ${cls}`}
                  >
                    {c}
                  </button>
                </li>
              )
            })}
          </ul>
          {pick.value !== null && (
            <div class="mt-3 text-sm">
              <p><strong>{pick.value === q().correct ? '✓ Correct.' : '✗ Not quite.'}</strong> {q().explain}</p>
              <button type="button" onClick={next} class="mt-2 text-accent-p1 underline">Next →</button>
            </div>
          )}
        </div>
      </div>
    </aside>
  )
}
