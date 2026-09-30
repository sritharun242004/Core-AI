import { createReadStream } from 'node:fs'
import { stat } from 'node:fs/promises'
import { extname, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

/** Serve the already-built Pagefind index in Astro dev; no copying generated assets. */
export function pagefindDev() {
  const root = fileURLToPath(new URL('./dist/client/pagefind/', import.meta.url))
  return {
    name: 'core-ai-pagefind-dev',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/pagefind', async (req, res, next) => {
        try {
          const relative = decodeURIComponent((req.url ?? '/').split('?')[0])
          const path = resolve(root, `.${relative}`)
          if (!path.startsWith(resolve(root) + sep)) {
            res.statusCode = 403
            res.end()
            return
          }
          if (!(await stat(path)).isFile()) {
            next()
            return
          }
          const types = {
            '.js': 'text/javascript',
            '.wasm': 'application/wasm',
            '.json': 'application/json',
            '.css': 'text/css',
          }
          res.setHeader('Content-Type', types[extname(path)] ?? 'application/octet-stream')
          res.setHeader('Cache-Control', 'no-cache')
          createReadStream(path)
            .on('error', () => res.destroy())
            .pipe(res)
        } catch {
          next()
        }
      })
    },
  }
}
