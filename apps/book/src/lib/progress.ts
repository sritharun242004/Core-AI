export interface AnswerStats { attempts: number; correct: number }
export interface Progress { answers: Record<string, AnswerStats> }

const KEY = 'core-ai:progress'

function safeParse(raw: string | null): Progress {
  if (!raw) return { answers: {} }
  try {
    const p = JSON.parse(raw)
    if (!p || typeof p !== 'object') return { answers: {} }
    // typeof null === 'object' — must guard for it explicitly.
    if (p.answers === null || typeof p.answers !== 'object') return { answers: {} }
    return p as Progress
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
  const cur = p.answers[id] ?? { attempts: 0, correct: 0 }
  p.answers[id] = { attempts: cur.attempts + 1, correct: cur.correct + (correct ? 1 : 0) }
  setProgress(p)
}

export function getMastery(id: string): number {
  const s = getProgress().answers[id]
  if (!s || s.attempts === 0) return 0
  return s.correct / s.attempts
}

// Surface-tagged progress helpers (used by /interview dashboard)
export interface SurfaceCounts { coding: number; sysdes: number; fundamentals: number; behavioral: number }
export type Surface = keyof SurfaceCounts

export function surfaceCounts(): SurfaceCounts {
  const p = getProgress()
  const out: SurfaceCounts = { coding: 0, sysdes: 0, fundamentals: 0, behavioral: 0 }
  for (const [id, s] of Object.entries(p.answers)) {
    if (s.correct === 0) continue
    if      (id.startsWith('coding:'))    out.coding++
    else if (id.startsWith('sysdes:'))    out.sysdes++
    else if (id.startsWith('beh:'))       out.behavioral++
    else                                  out.fundamentals++
  }
  return out
}

export function totalSolved(): number {
  return Object.values(getProgress().answers).filter((a) => a.correct > 0).length
}
