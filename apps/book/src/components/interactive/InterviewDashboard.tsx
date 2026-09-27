import { useSignal } from '@preact/signals'
import { useEffect } from 'preact/hooks'
import { getProgress, surfaceCounts, totalSolved } from '../../lib/progress'
import { COMPANIES } from '../../lib/companies'

export function InterviewDashboard() {
  const tick = useSignal(0)
  useEffect(() => {
    if (typeof window === 'undefined') return
    const bump = () => (tick.value = tick.value + 1)
    addEventListener('storage', bump)
    addEventListener('progress:changed', bump)
    return () => {
      removeEventListener('storage', bump)
      removeEventListener('progress:changed', bump)
    }
  }, [])

  // Reads happen at render time; tick.value in JSX forces re-subscription.
  const _sub = tick.value
  const total = totalSolved()
  const surfaces = surfaceCounts()
  const perCompany: Record<string, number> = {}
  {
    for (const c of COMPANIES) perCompany[c.slug] = 0
    for (const id of Object.keys(getProgress().answers)) {
      for (const c of COMPANIES) if (id.includes(`:${c.slug}`)) perCompany[c.slug]++
    }
  }

  return (
    <section class="my-6 grid gap-4 md:grid-cols-2" data-tick={_sub}>
      <div class="rounded-md border border-border-soft p-4">
        <p class="text-sm text-fg-muted">Total problems solved</p>
        <p class="font-display text-4xl mt-1">{total}</p>
      </div>
      <div class="rounded-md border border-border-soft p-4">
        <p class="text-sm text-fg-muted">By surface</p>
        <ul class="mt-2 text-sm space-y-1">
          {(['coding', 'sysdes', 'fundamentals', 'behavioral'] as const).map((k) => (
            <li class="flex justify-between"><span>{k}</span><span>{surfaces[k]}</span></li>
          ))}
        </ul>
      </div>
      <div class="rounded-md border border-border-soft p-4 md:col-span-2">
        <p class="text-sm text-fg-muted">By company</p>
        <ul class="mt-2 text-sm grid grid-cols-2 md:grid-cols-4 gap-2">
          {COMPANIES.map((c) => (
            <li class="flex justify-between border-b border-border-soft/50 py-1">
              <span>{c.emoji} {c.name}</span><span>{perCompany[c.slug]}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
