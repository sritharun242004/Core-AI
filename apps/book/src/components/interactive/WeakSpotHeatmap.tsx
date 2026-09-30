import { getMastery, getProgress } from '../../lib/progress'

export function WeakSpotHeatmap() {
  const ids = Object.keys(getProgress().answers).sort()
  if (ids.length === 0) {
    return (
      <p class="text-fg-muted text-sm italic">
        Answer some quiz questions first to see weak spots.
      </p>
    )
  }
  return (
    <section class="my-6">
      <p class="text-sm text-fg-muted mb-2">Darker = weaker (mastery below 0.5)</p>
      <div class="grid grid-cols-6 md:grid-cols-10 gap-1">
        {ids.map((id) => {
          const m = getMastery(id)
          const opacity = 0.15 + (1 - m) * 0.85
          return (
            <button
              type="button"
              key={id}
              title={`${id} — mastery ${Math.round(m * 100)}%`}
              class="aspect-square rounded-sm text-[10px] leading-none"
              style={{ background: `rgba(217, 87, 87, ${opacity})`, color: 'var(--fg)' }}
            >
              {id.slice(0, 6)}
            </button>
          )
        })}
      </div>
    </section>
  )
}
