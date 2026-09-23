import { defineCollection, z } from 'astro:content'
import { glob } from 'astro/loaders'

const weeks = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/weeks' }),
  schema: z.object({
    week: z.number().int().min(1).max(25),
    part: z.number().int().min(1).max(6),                 // Month 1..6
    slug: z.string(),                                     // e.g. "week-01-linear-algebra"
    title: z.string(),
    hook: z.string(),                                     // one striking sentence
    hours: z.number().default(20),
    computeTier: z.enum(['green', 'yellow', 'red']),
    difficulty: z.number().int().min(1).max(5),
    prereqSlugs: z.array(z.string()).default([]),
    referenceProject: z.string().optional(),              // path like "projects/week-01-linalg-lab"
    accentVar: z.string().default('--accent-p1'),
    publishedAt: z.date().optional()
  })
})

const companies = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/companies' }),
  schema: z.object({
    slug: z.enum(['openai', 'anthropic', 'deepmind', 'meta', 'xai', 'deepseek', 'qwen']),
    name: z.string(),
    tint: z.string(),                                     // css color for the pill
    tagline: z.string(),
    founded: z.number().int(),
    hq: z.string()
  })
})

export const collections = { weeks, companies }
