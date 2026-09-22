import { defineConfig } from '@playwright/test'
export default defineConfig({
  testDir: './tests',
  webServer: { command: 'pnpm build && pnpm preview --host 127.0.0.1 --port 4321', url: 'http://127.0.0.1:4321', reuseExistingServer: false, timeout: 60_000 },
  use: { baseURL: 'http://127.0.0.1:4321' }
})
