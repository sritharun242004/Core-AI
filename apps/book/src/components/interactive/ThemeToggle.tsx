import { signal, effect } from '@preact/signals'

const theme = signal<'light' | 'dark'>(
  typeof document !== 'undefined' && document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light',
)

if (typeof document !== 'undefined') {
  effect(() => {
    document.documentElement.dataset.theme = theme.value
    try { localStorage.setItem('theme', theme.value) } catch {}
  })
}

export function ThemeToggle() {
  return (
    <button
      type="button"
      aria-label="Toggle theme"
      onClick={() => (theme.value = theme.value === 'dark' ? 'light' : 'dark')}
      class="px-3 py-1 rounded-sm border border-border-soft text-sm"
    >
      {theme.value === 'dark' ? '☀️' : '🌙'}
    </button>
  )
}
