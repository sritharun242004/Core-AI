#!/usr/bin/env node
import { readFile, readdir } from 'node:fs/promises'
import { join } from 'node:path'

// Known-flaky or known-blocking URLs that resolve fine in a browser but fail
// HEAD requests from CI/CLI environments (bot-blocking, method not allowed, etc),
// plus placeholder/example URLs embedded in internal planning docs (not real
// hyperlinks readers would click). Add entries here instead of removing the
// link/snippet from content.
const IGNORE_URLS = new Set([
  // docs/superpowers/plans/2026-09-22-core-ai-foundation.md — local dev server
  // and placeholder domains used in code samples, never reachable from any host.
  'http://localhost:4321',
  'http://localhost:4321/$s', // shell-loop artifact in a historical plan
  'http://localhost:4321/$s`', // same artifact mentioned in the handoff
  'http://127.0.0.1:4321',
  'https://core-ai.book',
  'https://core-ai-<hash', // regex artifact of `https://core-ai-<hash>.vercel.app`
  // placeholder GitHub repo referenced in an example footer snippet
  'https://github.com/tharun/core-ai-book',
  'https://github.com/tharun/core-ai-book/tree/main/${path}`}', // JSX template-literal artifact
  // real Wikimedia asset; upload.wikimedia.org rejects HEAD (and even GET) with
  // 400 regardless of User-Agent — verified working link, checker-method quirk.
  'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg',
])

async function* mdxFiles(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue
    const p = join(dir, entry.name)
    if (entry.isDirectory()) yield* mdxFiles(p)
    else if (p.endsWith('.mdx') || p.endsWith('.md')) yield p
  }
}

const urlRe = /https?:\/\/[^\s)>\]"']+/g
const seen = new Set()
const failed = []

for await (const f of mdxFiles('.')) {
  const text = await readFile(f, 'utf-8')
  for (const m of text.matchAll(urlRe)) {
    const url = m[0].replace(/[.,;:!)\]]+$/, '')
    if (seen.has(url) || IGNORE_URLS.has(url)) continue
    seen.add(url)
    try {
      const r = await fetch(url, {
        method: 'HEAD',
        redirect: 'follow',
        signal: AbortSignal.timeout(8000),
      })
      if (!r.ok) failed.push({ url, status: r.status, file: f })
    } catch (e) {
      failed.push({ url, status: e.name, file: f })
    }
  }
}

if (failed.length) {
  console.error('BROKEN LINKS:')
  for (const x of failed) console.error(`  ${x.status}  ${x.url}   (in ${x.file})`)
  process.exit(1)
}
console.log(`OK — checked ${seen.size} unique URLs`)
