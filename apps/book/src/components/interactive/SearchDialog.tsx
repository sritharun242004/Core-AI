import { signal, effect } from '@preact/signals'

interface PagefindResult { id: string; data: () => Promise<{ url: string; excerpt: string; meta: { title: string } }> }
interface PagefindModule { search: (q: string) => Promise<{ results: PagefindResult[] }> }

const open = signal(false)
const query = signal('')
const hits = signal<{ url: string; title: string; excerpt: string }[]>([])
let pagefind: PagefindModule | null = null

async function ensurePagefind(): Promise<PagefindModule | null> {
  if (pagefind) return pagefind
  try {
    // Runtime-constructed path so Vite/Rollup won't try to resolve it at build.
    const url = new URL('/pagefind/pagefind.js', location.origin).href
    // @ts-expect-error — dynamic import of runtime asset
    pagefind = await import(/* @vite-ignore */ url)
    return pagefind
  } catch {
    return null
  }
}

async function runSearch(q: string) {
  if (!q) { hits.value = []; return }
  const pf = await ensurePagefind()
  if (!pf) { hits.value = []; return }
  const res = await pf.search(q)
  const first = res.results.slice(0, 8)
  const data = await Promise.all(first.map((r) => r.data()))
  hits.value = data.map((d) => ({ url: d.url, title: d.meta.title, excerpt: d.excerpt }))
}

if (typeof window !== 'undefined') {
  addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); open.value = true }
    if (e.key === 'Escape') open.value = false
  })
  effect(() => { runSearch(query.value) })
}

export function SearchDialog() {
  if (!open.value) return null
  return (
    <div role="dialog" aria-modal="true" aria-label="Search" class="fixed inset-0 z-50 bg-black/40 flex items-start justify-center p-8" onClick={() => (open.value = false)}>
      <div class="bg-canvas w-full max-w-xl rounded-md border border-border-soft p-4" onClick={(e) => e.stopPropagation()}>
        <input
          autoFocus
          type="search"
          value={query.value}
          onInput={(e) => (query.value = (e.target as HTMLInputElement).value)}
          placeholder="Search the book…"
          class="w-full bg-canvas-subtle text-fg px-3 py-2 rounded-sm border border-border-soft"
        />
        <ul class="mt-3 space-y-2">
          {hits.value.map((h) => (
            <li>
              <a href={h.url} class="block px-3 py-2 rounded-sm hover:bg-canvas-subtle">
                <p class="font-medium">{h.title}</p>
                <p class="text-sm text-fg-muted" dangerouslySetInnerHTML={{ __html: h.excerpt }} />
              </a>
            </li>
          ))}
          {query.value && hits.value.length === 0 && (
            <li class="text-fg-muted text-sm italic">No matches.</li>
          )}
        </ul>
        <p class="text-xs text-fg-muted mt-2">⌘K to open · Esc to close</p>
      </div>
    </div>
  )
}

export function SearchOpener() {
  return (
    <button type="button" onClick={() => (open.value = true)} class="text-sm px-2 py-1 rounded-sm border border-border-soft">
      ⌘K Search
    </button>
  )
}
