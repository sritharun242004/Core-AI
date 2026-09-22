import { defineConfig, fontProviders } from 'astro/config'
import mdx from '@astrojs/mdx'
import preact from '@astrojs/preact'
import sitemap from '@astrojs/sitemap'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  site: 'https://core-ai.book',
  integrations: [mdx(), preact({ compat: false }), sitemap()],
  vite: { plugins: [tailwindcss()] },
  fonts: [
    { name: 'Fraunces', cssVariable: '--font-display', provider: fontProviders.google(), weights: [400, 600, 800], styles: ['normal', 'italic'] },
    { name: 'Inter', cssVariable: '--font-body', provider: fontProviders.google(), weights: [400, 500, 600, 700], subsets: ['latin'] },
    { name: 'JetBrains Mono', cssVariable: '--font-mono', provider: fontProviders.google(), weights: [400, 600] }
  ],
  markdown: {
    shikiConfig: { theme: 'github-dark-dimmed', wrap: false }
  }
})
