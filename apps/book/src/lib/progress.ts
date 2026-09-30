import { type Sm2Result, nextDueDate } from './quiz'

export interface AnswerStats extends Partial<Sm2Result> {
  attempts: number
  correct: number
}
export interface Progress {
  answers: Record<string, AnswerStats>
}

const KEY = 'core-ai:progress'

function safeParse(raw: string | null): Progress {
  if (!raw) return { answers: {} }
  try {
    const p = JSON.parse(raw)
    if (!p || typeof p !== 'object') return { answers: {} }
    // typeof null === 'object' — must guard for it explicitly.
    if (p.answers === null || typeof p.answers !== 'object' || Array.isArray(p.answers))
      return { answers: {} }
    const answers: Record<string, AnswerStats> = {}
    for (const [id, value] of Object.entries(p.answers)) {
      if (!value || typeof value !== 'object') continue
      const s = value as AnswerStats
      if (
        !Number.isInteger(s.attempts) ||
        !Number.isInteger(s.correct) ||
        s.attempts < 0 ||
        s.correct < 0 ||
        s.correct > s.attempts
      )
        continue
      const answer: AnswerStats = { attempts: s.attempts, correct: s.correct }
      if (
        Number.isInteger(s.reps) &&
        (s.reps ?? -1) >= 0 &&
        Number.isFinite(s.ease) &&
        (s.ease ?? 0) >= 1.3 &&
        (s.ease ?? 3) <= 2.8 &&
        Number.isInteger(s.interval) &&
        (s.interval ?? -1) >= 0 &&
        Number.isFinite(s.dueDate)
      ) {
        Object.assign(answer, {
          reps: s.reps,
          ease: s.ease,
          interval: s.interval,
          dueDate: s.dueDate,
        })
      }
      Object.defineProperty(answers, id, {
        value: answer,
        enumerable: true,
        writable: true,
        configurable: true,
      })
    }
    return { answers }
  } catch {
    return { answers: {} }
  }
}

export function getProgress(): Progress {
  if (typeof localStorage === 'undefined') return { answers: {} }
  try {
    return safeParse(localStorage.getItem(KEY))
  } catch {
    return { answers: {} }
  }
}

export function setProgress(p: Progress): void {
  if (typeof localStorage === 'undefined') return
  try {
    localStorage.setItem(KEY, JSON.stringify(p))
  } catch {
    // private mode, quota — silently drop
  }
}

export function recordAnswer(id: string, correct: boolean): void {
  const p = getProgress()
  const cur = Object.hasOwn(p.answers, id) ? p.answers[id] : { attempts: 0, correct: 0 }
  Object.defineProperty(p.answers, id, {
    value: { ...cur, attempts: cur.attempts + 1, correct: cur.correct + (correct ? 1 : 0) },
    enumerable: true,
    writable: true,
    configurable: true,
  })
  setProgress(p)
}

export function recordQuizAnswer(id: string, correct: boolean, now = Date.now()): Sm2Result {
  recordAnswer(id, correct)
  const p = getProgress()
  const cur = Object.hasOwn(p.answers, id)
    ? p.answers[id]
    : { attempts: 1, correct: correct ? 1 : 0 }
  const schedule = nextDueDate(
    { reps: cur.reps ?? 0, ease: cur.ease ?? 2.5, interval: cur.interval ?? 0 },
    correct,
    now,
  )
  Object.defineProperty(p.answers, id, {
    value: { ...cur, ...schedule },
    enumerable: true,
    writable: true,
    configurable: true,
  })
  setProgress(p)
  return schedule
}

export function getMastery(id: string): number {
  const s = getProgress().answers[id]
  if (!s || s.attempts === 0) return 0
  return s.correct / s.attempts
}

// Surface-tagged progress helpers (used by /interview dashboard)
export interface SurfaceCounts {
  coding: number
  sysdes: number
  fundamentals: number
  behavioral: number
}
export type Surface = keyof SurfaceCounts

export function surfaceCounts(): SurfaceCounts {
  const p = getProgress()
  const out: SurfaceCounts = { coding: 0, sysdes: 0, fundamentals: 0, behavioral: 0 }
  for (const [id, s] of Object.entries(p.answers)) {
    if (s.correct === 0) continue
    if (id.startsWith('coding:')) out.coding++
    else if (id.startsWith('sysdes:')) out.sysdes++
    else if (id.startsWith('beh:')) out.behavioral++
    else out.fundamentals++
  }
  return out
}

export function totalSolved(): number {
  return Object.values(getProgress().answers).filter((a) => a.correct > 0).length
}
