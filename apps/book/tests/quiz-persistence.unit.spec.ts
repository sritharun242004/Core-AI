/** @vitest-environment jsdom */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getProgress, recordAnswer, getMastery, setProgress } from '../src/lib/progress'

describe('progress store', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
  })

  it('returns a fresh empty progress when localStorage is empty', () => {
    expect(getProgress()).toEqual({ answers: {} })
  })

  it('records answers and computes mastery in [0, 1]', () => {
    recordAnswer('q1', true)
    recordAnswer('q1', true)
    recordAnswer('q1', false)
    const m = getMastery('q1')
    expect(m).toBeGreaterThan(0)
    expect(m).toBeLessThanOrEqual(1)
  })

  it('survives malformed JSON in localStorage without throwing', () => {
    localStorage.setItem('core-ai:progress', '{{{ not json')
    expect(() => getProgress()).not.toThrow()
    expect(getProgress()).toEqual({ answers: {} })
  })

  it('survives localStorage.setItem throwing (private mode)', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('QuotaExceeded')
    })
    expect(() => setProgress({ answers: { q1: { attempts: 1, correct: 1 } } })).not.toThrow()
  })
})
