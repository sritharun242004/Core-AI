import { useComputed, useSignal } from '@preact/signals'
import { recordQuizAnswer } from '../../lib/progress'
import type { RecallQuestion } from './MicroRecall'

export interface WeeklyQuizProps {
  weekId: number
  questions: RecallQuestion[]
}

export function WeeklyQuiz({ weekId, questions }: WeeklyQuizProps) {
  const idx = useSignal(0)
  const answers = useSignal<{ correct: boolean; days: number }[]>([])
  const pick = useSignal<number | null>(null)
  const finished = useSignal(false)
  const score = useComputed(() => answers.value.filter((answer) => answer.correct).length)

  const submit = (choice: number) => {
    if (pick.value !== null || finished.value) return
    const question = questions[idx.value]
    pick.value = choice
    const correct = choice === question.correct
    const schedule = recordQuizAnswer(question.id, correct)
    answers.value = [...answers.value, { correct, days: schedule.interval }]
    if (typeof window !== 'undefined') window.dispatchEvent(new Event('progress:changed'))
  }
  const next = () => {
    if (idx.value === questions.length - 1) finished.value = true
    else {
      idx.value += 1
      pick.value = null
    }
  }

  if (!questions.length)
    return <p class="my-6 text-fg-muted">No review questions in this lesson.</p>
  if (finished.value) {
    return (
      <section class="my-10 border border-border-soft rounded-md p-6" aria-live="polite">
        <h2 class="font-display text-2xl">
          Weekly synthesis — score {score.value} / {questions.length}
        </h2>
        <ul class="mt-4 space-y-1 text-sm">
          {answers.value.map((answer, index) => (
            <li key={questions[index].id}>
              Q{index + 1}: {answer.correct ? 'Correct' : 'Review'} — next due in{' '}
              <strong>{answer.days}</strong> day{answer.days !== 1 ? 's' : ''}
            </li>
          ))}
        </ul>
        <p class="mt-3 text-sm text-fg-muted">Review history is stored on this browser only.</p>
      </section>
    )
  }

  const question = questions[idx.value]
  const revealed = pick.value !== null
  return (
    <section
      class="my-10 border border-border-soft rounded-md p-6"
      aria-label={`Week ${weekId} review`}
    >
      <p class="text-fg-muted text-sm">
        Question {idx.value + 1} of {questions.length}
      </p>
      <p class="font-display text-lg mt-2">{question.prompt}</p>
      <ul class="mt-3 space-y-2">
        {question.choices.map((choice, index) => {
          const chosen = pick.value === index
          const right = index === question.correct
          const state = !revealed
            ? 'border-border-soft'
            : right
              ? 'border-accent-p5 bg-accent-p5/10'
              : chosen
                ? 'border-accent-p6 bg-accent-p6/10'
                : 'border-border-soft'
          return (
            <li key={`${question.id}-${index}`}>
              <button
                type="button"
                disabled={revealed}
                onClick={() => submit(index)}
                class={`w-full text-left px-3 py-2 rounded-sm border ${state}`}
              >
                {choice}
                {revealed && right
                  ? ' — correct answer'
                  : revealed && chosen
                    ? ' — your answer'
                    : ''}
              </button>
            </li>
          )
        })}
      </ul>
      {revealed && (
        <div class="mt-3 text-sm" aria-live="polite">
          <p>{question.explain}</p>
          <button type="button" onClick={next} class="mt-2 text-accent-p1 underline py-2">
            {idx.value === questions.length - 1 ? 'View results' : 'Next question →'}
          </button>
        </div>
      )}
    </section>
  )
}
