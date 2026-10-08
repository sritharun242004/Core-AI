import mdx from '@astrojs/mdx'
import preact from '@astrojs/preact'
import sitemap from '@astrojs/sitemap'
import vercel from '@astrojs/vercel'
import tailwindcss from '@tailwindcss/vite'
import expressiveCode from 'astro-expressive-code'
import { defineConfig, fontProviders } from 'astro/config'
import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'
import { pagefindDev } from './pagefind-dev.mjs'

export default defineConfig({
  site: 'https://core-ai.book',
  output: 'static',
  adapter: vercel(),
  integrations: [
    expressiveCode({
      themes: ['github-light', 'github-dark-dimmed'],
      styleOverrides: { codeFontFamily: 'var(--font-mono)' },
    }),
    mdx(),
    preact({ compat: true }),
    sitemap(),
  ],
  vite: {
    plugins: [tailwindcss(), pagefindDev()],
    resolve: {
      alias: {
        'react/jsx-runtime': 'preact/jsx-runtime',
        react: 'preact/compat',
        'react-dom/test-utils': 'preact/test-utils',
        'react-dom': 'preact/compat',
      },
    },
    ssr: {
      noExternal: ['motion', 'framer-motion'],
    },
  },
  fonts: [
    {
      name: 'Titan One',
      cssVariable: '--font-display',
      provider: fontProviders.google(),
      weights: [400],
      subsets: ['latin'],
      fallbacks: ['Arial Black', 'Impact', 'system-ui', 'sans-serif'],
    },
    {
      name: 'Nunito',
      cssVariable: '--font-body',
      provider: fontProviders.google(),
      weights: [400, 600, 700, 800, 900],
      subsets: ['latin'],
      fallbacks: ['Avenir Next', 'Segoe UI', 'system-ui', 'sans-serif'],
    },
    {
      name: 'JetBrains Mono',
      cssVariable: '--font-mono',
      provider: fontProviders.google(),
      weights: [400, 500, 600],
    },
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, { output: 'html', strict: 'warn' }]],
    shikiConfig: { theme: 'github-dark-dimmed', wrap: false },
  },
})
