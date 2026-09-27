/** @vitest-environment jsdom */
import { render, fireEvent } from '@testing-library/preact'
import { beforeEach, describe, expect, it } from 'vitest'
import { MicroRecall } from '../src/components/interactive/MicroRecall'

describe('MicroRecall reveal state', () => {
  beforeEach(() => { localStorage.clear() })

  it('shows the explanation after the user picks a choice', async () => {
    const questions = [{
      id: 'q-a',
      prompt: 'What is 2+2?',
      choices: ['3', '4', '5'],
      correct: 1,
      explain: 'Addition is closed on the integers.',
    }]
    const { container, getByText } = render(<MicroRecall questions={questions} />)
    // Click the correct answer.
    const btn = getByText('4') as HTMLButtonElement
    fireEvent.click(btn)
    // After the pick, the explanation and a Next button must be visible.
    // With signal() inside the component body, a re-render would recreate `pick`
    // as null and the reveal branch would never surface.
    expect(container.textContent).toContain('Addition is closed on the integers.')
    expect(container.textContent).toContain('Next')
  })
})
