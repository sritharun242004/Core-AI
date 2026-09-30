/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen } from '@testing-library/preact'
import { afterEach, beforeEach, expect, it } from 'vitest'
import { WeeklyQuiz } from '../src/components/interactive/WeeklyQuiz'
import { getProgress, recordQuizAnswer } from '../src/lib/progress'

beforeEach(() => localStorage.clear())
afterEach(cleanup)

it('persists the actual question id and shows final feedback before the summary', () => {
  render(
    <WeeklyQuiz
      weekId={22}
      questions={[
        {
          id: '22L-tools',
          prompt: 'Which boundary?',
          choices: ['Allowlist', 'Anything'],
          correct: 0,
          explain: 'Only approved tools may run.',
        },
      ]}
    />,
  )
  fireEvent.click(screen.getByRole('button', { name: 'Allowlist' }))
  expect(screen.getByText('Only approved tools may run.')).toBeTruthy()
  expect(getProgress().answers['22L-tools'].correct).toBe(1)
  expect(getProgress().answers['week22-q0']).toBeUndefined()
  fireEvent.click(screen.getByRole('button', { name: /results/i }))
  expect(screen.getByText(/score 1 \/ 1/)).toBeTruthy()
})

it('uses previous spaced-review state without colliding across tracks', () => {
  const first = recordQuizAnswer('22L-q0', true, 0)
  const second = recordQuizAnswer('22L-q0', true, 1000)
  recordQuizAnswer('22P-q0', false, 1000)
  expect(first.interval).toBe(1)
  expect(second.interval).toBe(3)
  expect(getProgress().answers['22L-q0'].attempts).toBe(2)
  expect(getProgress().answers['22P-q0'].attempts).toBe(1)
  expect(second.dueDate).toBe(1000 + 3 * 86400000)
})

it('handles an empty quiz without indexing an absent question', () => {
  render(<WeeklyQuiz weekId={25} questions={[]} />)
  expect(screen.getByText(/No review questions/)).toBeTruthy()
})
