import { signal } from '@preact/signals'
import { getProgress, surfaceCounts, totalSolved } from '../../lib/progress'
import { COMPANIES } from '../../lib/companies'

export function InterviewDashboard() {
  const tick = signal(0)
  if (typeof window !== 'undefined') {
    const onStorage = () => (tick.value = tick.value + 1)
    addEventListener('storage', onStorage)
  }

  const total = () => (tick.value, totalSolved())
  const surfaces = () => (tick.value, surfaceCounts())
  const perCompany = () => {
    tick.value
    const p = getProgress()
    const map: Record<string, number> = {}
    for (const c of COMPANIES) map[c.slug] = 0
    for (const id of Object.keys(p.answers)) {
      for (const c of COMPANIES) if (id.includes(`:${c.slug}`)) map[c.slug]++
    }
    return map
  }

  return (
    <section class="my-6 grid gap-4 md:grid-cols-2">
      <div class="rounded-md border border-border-soft p-4">
        <p class="text-sm text-fg-muted">Total problems solved</p>
        <p class="font-display text-4xl mt-1">{total()}</p>
      </div>
      <div class="rounded-md border border-border-soft p-4">
        <p class="text-sm text-fg-muted">By surface</p>
        <ul class="mt-2 text-sm space-y-1">
          {(['coding', 'sysdes', 'fundamentals', 'behavioral'] as const).map((k) => (
            <li class="flex justify-between"><span>{k}</span><span>{surfaces()[k]}</span></li>
          ))}
        </ul>
      </div>
      <div class="rounded-md border border-border-soft p-4 md:col-span-2">
        <p class="text-sm text-fg-muted">By company</p>
        <ul class="mt-2 text-sm grid grid-cols-2 md:grid-cols-4 gap-2">
          {COMPANIES.map((c) => (
            <li class="flex justify-between border-b border-border-soft/50 py-1">
              <span>{c.emoji} {c.name}</span><span>{perCompany()[c.slug]}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
