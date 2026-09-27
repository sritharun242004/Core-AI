import { defineConfig } from 'vitest/config'
import preact from '@preact/preset-vite'

export default defineConfig({
  plugins: [preact()],
  test: {
    environment: 'jsdom',
    globals: false,
    include: ['tests/**/*.spec.tsx', 'tests/**/*.unit.spec.ts'],
  },
})
