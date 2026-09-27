import { defineCollection, z } from 'astro:content'
import { glob } from 'astro/loaders'

const weeks = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/weeks' }),
  schema: z.object({
    week: z.number().int().min(1).max(25),
    part: z.number().int().min(1).max(6),
    slug: z.string(),
    title: z.string(),
    hook: z.string(),
    hours: z.number().default(20),
    computeTier: z.enum(['green', 'yellow', 'red']),
    difficulty: z.number().int().min(1).max(5),
    prereqSlugs: z.array(z.string()).default([]),
    referenceProject: z.string().optional(),
    accentVar: z.string().default('--accent-p1'),
    publishedAt: z.date().optional(),
  }),
})

const companies = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/companies' }),
  schema: z.object({
    slug: z.enum(['openai', 'anthropic', 'deepmind', 'meta', 'xai', 'deepseek', 'qwen']),
    name: z.string(),
    tint: z.string(),
    tagline: z.string(),
    founded: z.number().int(),
    hq: z.string(),
    philosophyLead: z.string(),
    leaders: z.array(z.string()).default([]),
    modelTimeline: z
      .array(z.object({ year: z.number().int(), note: z.string() }))
      .default([]),
    loop: z
      .array(z.object({ round: z.string(), note: z.string() }))
      .default([]),
    papers: z.array(z.string()).default([]),
    blogs: z.array(z.string()).default([]),
    books: z.array(z.string()).default([]),
    seededAngles: z
      .array(z.object({ week: z.number().int(), note: z.string() }))
      .default([]),
  }),
})

const extras = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/extras' }),
  schema: z.object({
    slug: z.enum(['how-to-study', 'glossary', 'math-primer', 'paper-reading-protocol', 'tech-writing']),
    title: z.string(),
    tagline: z.string(),
  }),
})

export const collections = { weeks, companies, extras }
