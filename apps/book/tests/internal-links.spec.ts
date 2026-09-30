import { existsSync } from 'node:fs'
import { readFile, readdir } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { expect, test } from '@playwright/test'

test('all emitted pages link to existing local routes or declared server routes', async () => {
  const root = resolve(process.cwd(), 'dist/client')
  const serverRoutes = new Set(['/companies/compare', '/motion-probe'])
  async function inspect(directory: string) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name)
      if (entry.isDirectory()) {
        if (entry.name !== 'pagefind') await inspect(path)
      } else if (entry.name.endsWith('.html')) {
        const html = await readFile(path, 'utf8')
        for (const [, href] of html.matchAll(/href="(\/[^"#]*)"/g)) {
          if (href.startsWith('//')) continue
          const pathname = new URL(href.replaceAll('&amp;', '&'), 'http://localhost').pathname
          const normalized = pathname.replace(/\/$/, '') || '/'
          if (serverRoutes.has(normalized)) continue
          const target = join(root, pathname)
          expect
            .soft(existsSync(target) || existsSync(join(target, 'index.html')), `${path}: ${href}`)
            .toBe(true)
        }
      }
    }
  }
  await inspect(root)
})
