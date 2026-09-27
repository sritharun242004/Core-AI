import { defineConfig } from '@playwright/test'
export default defineConfig({
  testDir: './tests',
  testIgnore: ['**/*.unit.spec.ts', '**/*.spec.tsx'],
  workers: 1,
  webServer: {
    command: 'pnpm build && pnpm dev --host 127.0.0.1 --port 4321',
    url: 'http://127.0.0.1:4321',
    reuseExistingServer: false,
    timeout: 120_000,
  },
  use: { baseURL: 'http://127.0.0.1:4321' },
})
