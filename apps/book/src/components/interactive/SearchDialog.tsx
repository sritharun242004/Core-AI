import { effect, signal } from '@preact/signals'
import { useEffect, useRef } from 'preact/hooks'

interface PagefindResult {
  data: () => Promise<{ url: string; excerpt: string; meta: { title: string } }>
}
interface PagefindModule {
  search: (q: string) => Promise<{ results: PagefindResult[] }>
}

const open = signal(false)
const query = signal('')
const hits = signal<{ url: string; title: string; excerpt: string }[]>([])
const status = signal<'idle' | 'loading' | 'ready' | 'error'>('idle')
let pagefind: PagefindModule | null = null
let request = 0

async function runSearch(value: string) {
  const current = ++request
  const text = value.trim()
  if (!text) {
    hits.value = []
    status.value = 'idle'
    return
  }
  status.value = 'loading'
  hits.value = []
  try {
    if (!pagefind) {
      const url = new URL('/pagefind/pagefind.js', location.origin).href
      pagefind = (await import(/* @vite-ignore */ url)) as PagefindModule
    }
    const result = await pagefind.search(text)
    const data = await Promise.all(result.results.slice(0, 8).map((entry) => entry.data()))
    if (current !== request) return
    hits.value = data.map((entry) => ({
      url: entry.url,
      title: entry.meta.title,
      excerpt: new DOMParser().parseFromString(entry.excerpt, 'text/html').body.textContent ?? '',
    }))
    status.value = 'ready'
  } catch {
    if (current === request) {
      pagefind = null
      status.value = 'error'
    }
  }
}

if (typeof window !== 'undefined') {
  addEventListener('keydown', (event) => {
    if (event.key === 'Escape') open.value = false
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault()
      open.value = true
    }
  })
  effect(() => {
    void runSearch(query.value)
  })
  ;(window as unknown as { openSearch?: () => void }).openSearch = () => {
    open.value = true
  }
}

export function SearchDialog() {
  const ref = useRef<HTMLDialogElement>(null)
  const isOpen = open.value
  useEffect(() => {
    if (isOpen && !ref.current?.open) {
      ref.current?.showModal()
      ref.current?.querySelector('input')?.focus()
    } else if (!isOpen && ref.current?.open) ref.current.close()
  }, [isOpen])

  return (
    <dialog
      ref={ref}
      aria-label="Search the book"
      onClose={() => {
        open.value = false
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) open.value = false
      }}
      onKeyDown={(event) => {
        if (event.key === 'Escape') open.value = false
      }}
      class="m-auto mt-16 w-[calc(100%_-_2rem)] max-w-xl max-h-[80vh] overflow-y-auto bg-canvas text-fg rounded-md border border-border-soft p-5 backdrop:bg-black/40"
    >
      <div class="flex justify-between items-center gap-4 mb-4">
        <h2 class="font-display text-xl">Search the book</h2>
        <button
          type="button"
          onClick={() => {
            open.value = false
          }}
          class="px-3 py-2 border border-border-soft rounded-sm"
        >
          Close
        </button>
      </div>
      <label for="book-search" class="sr-only">
        Search query
      </label>
      <input
        id="book-search"
        type="search"
        maxLength={160}
        value={query.value}
        onInput={(event) => {
          query.value = event.currentTarget.value
        }}
        placeholder="Search concepts, lessons, or companies…"
        class="w-full bg-canvas-subtle text-fg px-3 py-3 rounded-sm border border-border-soft"
      />
      <div aria-live="polite" class="mt-3 text-sm text-fg-muted">
        {status.value === 'loading' && <p>Searching…</p>}
        {status.value === 'idle' && <p>Enter a topic to find it in the book.</p>}
        {status.value === 'error' && (
          <p>
            Search is unavailable. Check your connection or build the local index.
            <button
              type="button"
              class="block underline py-2"
              onClick={() => {
                void runSearch(query.value)
              }}
            >
              Retry search
            </button>
          </p>
        )}
        {status.value === 'ready' && hits.value.length === 0 && (
          <p>No matches. Try a broader topic.</p>
        )}
      </div>
      <ul class="mt-3 space-y-2">
        {hits.value.map((hit) => (
          <li key={hit.url}>
            <a href={hit.url} class="block px-3 py-2 rounded-sm hover:bg-canvas-subtle">
              <p class="font-medium">{hit.title}</p>
              <p class="text-sm text-fg-muted">{hit.excerpt}</p>
            </a>
          </li>
        ))}
      </ul>
      <p class="text-sm text-fg-muted mt-4">⌘K / Ctrl+K to open · Esc to close</p>
    </dialog>
  )
}

export function SearchOpener() {
  return (
    <button
      type="button"
      onClick={() => {
        open.value = true
      }}
      class="text-sm px-3 py-2 rounded-sm border border-border-soft"
    >
      Search
    </button>
  )
}
