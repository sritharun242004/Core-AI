import { defineConfig, fontProviders } from 'astro/config'
import expressiveCode from 'astro-expressive-code'
import mdx from '@astrojs/mdx'
import preact from '@astrojs/preact'
import sitemap from '@astrojs/sitemap'
import vercel from '@astrojs/vercel'
import tailwindcss from '@tailwindcss/vite'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

export default defineConfig({
  site: 'https://core-ai.book',
  output: 'static',
  adapter: vercel(),
  integrations: [
    expressiveCode({ themes: ['github-light', 'github-dark-dimmed'], styleOverrides: { codeFontFamily: 'var(--font-mono)' } }),
    mdx(),
    preact({ compat: false }),
    sitemap()
  ],
  vite: { plugins: [tailwindcss()] },
  fonts: [
    { name: 'Fraunces', cssVariable: '--font-display', provider: fontProviders.google(), weights: [400, 600, 800], styles: ['normal', 'italic'] },
    { name: 'Inter', cssVariable: '--font-body', provider: fontProviders.google(), weights: [400, 500, 600, 700], subsets: ['latin'] },
    { name: 'JetBrains Mono', cssVariable: '--font-mono', provider: fontProviders.google(), weights: [400, 600] }
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, { output: 'html', strict: 'warn' }]],
    shikiConfig: { theme: 'github-dark-dimmed', wrap: false }
  }
})
