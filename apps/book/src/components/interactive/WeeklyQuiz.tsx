import { useSignal, useComputed } from '@preact/signals'
import { recordAnswer } from '../../lib/progress'
import { nextDueDate, type Sm2State } from '../../lib/quiz'
import type { RecallQuestion } from './MicroRecall'

export interface WeeklyQuizProps { weekId: number; questions: RecallQuestion[] }

export function WeeklyQuiz({ weekId, questions }: WeeklyQuizProps) {
  const idx = useSignal(0)
  const answers = useSignal<{ i: number; correct: boolean }[]>([])
  const pick = useSignal<number | null>(null)

  const done = useComputed(() => answers.value.length === questions.length)
  const score = useComputed(() => answers.value.filter((a) => a.correct).length)

  const submit = (i: number) => {
    pick.value = i
    const correct = i === questions[idx.value].correct
    answers.value = [...answers.value, { i, correct }]
    recordAnswer(`week${weekId}-q${idx.value}`, correct)
    if (typeof window !== 'undefined') dispatchEvent(new Event('progress:changed'))
  }
  const next = () => {
    idx.value = idx.value + 1
    pick.value = null
  }

  if (done.value) {
    const summary = answers.value.map((a, i) => {
      const cur: Sm2State = { reps: a.correct ? 1 : 0, ease: 2.5, interval: 1 }
      const r = nextDueDate(cur, a.correct)
      return { i, days: r.interval, correct: a.correct }
    })
    return (
      <section class="my-10 border border-border-soft rounded-md p-6">
        <h2 class="font-display text-2xl">Weekly synthesis — score {score.value} / {questions.length}</h2>
        <ul class="mt-4 space-y-1 text-sm">
          {summary.map((s) => (
            <li>
              Q{s.i + 1}: {s.correct ? '✓' : '✗'} — next due in <strong>{s.days}</strong> day{s.days === 1 ? '' : 's'}
            </li>
          ))}
        </ul>
      </section>
    )
  }

  const q = questions[idx.value]
  const revealed = pick.value !== null
  return (
    <section class="my-10 border border-border-soft rounded-md p-6">
      <p class="text-fg-muted text-sm">Question {idx.value + 1} of {questions.length}</p>
      <p class="font-display text-lg mt-2">{q.prompt}</p>
      <ul class="mt-3 space-y-2">
        {q.choices.map((c, i) => {
          const chosen = pick.value === i
          const isRight = i === q.correct
          const cls =
            !revealed  ? 'border-border-soft' :
            isRight    ? 'border-accent-p5 bg-accent-p5/10' :
            chosen     ? 'border-accent-p6 bg-accent-p6/10' :
                         'border-border-soft opacity-50'
          return (
            <li>
              <button type="button" disabled={revealed} onClick={() => submit(i)}
                class={`w-full text-left px-3 py-2 rounded-sm border ${cls}`}>
                {c}
              </button>
            </li>
          )
        })}
      </ul>
      {revealed && (
        <div class="mt-3 text-sm">
          <p>{q.explain}</p>
          <button type="button" onClick={next} class="mt-2 text-accent-p1 underline">Next →</button>
        </div>
      )}
    </section>
  )
}
