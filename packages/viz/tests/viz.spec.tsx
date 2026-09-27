/** @vitest-environment jsdom */
import { render } from '@testing-library/preact'
import { describe, expect, it } from 'vitest'
import { VectorPlayground, MatrixMul } from '../src'

describe('VectorPlayground', () => {
  it('renders one <line> per input vector', () => {
    const { container } = render(
      <VectorPlayground vectors={[{ x: 3, y: 2 }, { x: -1, y: 4 }]} />,
    )
    expect(container.querySelectorAll('line').length).toBeGreaterThanOrEqual(2)
  })

  it('labels vectors when a label is provided', () => {
    const { container } = render(
      <VectorPlayground vectors={[{ x: 1, y: 1, label: 'v₁' }]} />,
    )
    expect(container.textContent).toContain('v₁')
  })
})

describe('MatrixMul', () => {
  it('renders every cell of A and B', () => {
    const { container } = render(
      <MatrixMul a={[[1, 2], [3, 4]]} b={[[5, 6], [7, 8]]} />,
    )
    expect(container.querySelectorAll('[data-cell]').length).toBeGreaterThanOrEqual(8)
  })
})
