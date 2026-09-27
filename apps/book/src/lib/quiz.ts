export interface Sm2State { reps: number; ease: number; interval: number }

export interface Sm2Result extends Sm2State { dueDate: number }

// SM-2-lite: reps=# consecutive correct; ease in [1.3, 2.8]; interval in days.
export function nextDueDate(cur: Sm2State, correct: boolean, now = Date.now()): Sm2Result {
  const reps = correct ? cur.reps + 1 : 0
  const ease = Math.max(1.3, Math.min(2.8, cur.ease + (correct ? 0.1 : -0.2)))
  const interval =
    !correct   ? 1 :
    reps === 1 ? 1 :
    reps === 2 ? 3 :
                 Math.round(cur.interval * ease)
  return { reps, ease, interval, dueDate: now + interval * 86_400_000 }
}
