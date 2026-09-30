import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const root = resolve(process.cwd(), '../..')
const directory = resolve(process.cwd(), 'src/content/weeks')
const files = readdirSync(directory).filter((file) => file.endsWith('.mdx'))
const slugs = files.map((file) => file.replace(/\.mdx$/, ''))

describe('complete curriculum inventory', () => {
  it('ships every track choice, capstone and final interview lesson without Week 16', () => {
    expect(files).toHaveLength(29)
    for (const slug of [
      'week-22l-agents-lab',
      'week-23l-mcp-a2a-adk-lab',
      'week-22p-two-tower-recsys',
      'week-23p-learning-to-rank-timeseries',
      'week-22r-transformer-repro',
      'week-23r-scaling-dpo-repro',
      'week-24-capstone',
      'week-25-interview-prep',
    ])
      expect(slugs).toContain(slug)
    expect(slugs.some((slug) => slug.startsWith('week-16-'))).toBe(false)
  })

  it('every referenced prerequisite and project path exists', () => {
    for (const file of files) {
      const source = readFileSync(resolve(directory, file), 'utf8')
      const frontmatter = source.split('---')[1]
      const prerequisites = frontmatter.match(/prereqSlugs:\s*\[([^\]]*)\]/)?.[1] ?? ''
      for (const [, slug] of prerequisites.matchAll(/["']([^"']+)["']/g)) {
        expect(slugs, `${file}: prerequisite ${slug}`).toContain(slug)
      }
      const project = frontmatter.match(/referenceProject:\s*["']([^"']+)["']/)?.[1]
      if (project) expect(existsSync(resolve(root, project)), `${file}: ${project}`).toBe(true)
    }
  })

  it('all 27 Python packages include runnable teaching artifacts', () => {
    const projects = readdirSync(resolve(root, 'projects')).filter((name) =>
      name.startsWith('week-'),
    )
    expect(projects).toHaveLength(27)
    for (const project of projects) {
      for (const item of [
        'pyproject.toml',
        'README.md',
        'SOLUTION_NOTES.md',
        'COMPUTE.md',
        'src',
        'tests',
        'notebooks',
        'assignments/warmup.md',
        'assignments/build.md',
        'assignments/challenge.md',
      ]) {
        expect(existsSync(resolve(root, 'projects', project, item)), `${project}/${item}`).toBe(
          true,
        )
      }
    }
    for (const track of ['research', 'applied-ml', 'llm-product']) {
      expect(existsSync(resolve(root, `capstone/track-${track}/RUBRIC.md`))).toBe(true)
    }
  })

  it('review question IDs are unique across lesson sources', () => {
    const owners = new Map<string, string>()
    for (const file of files) {
      const source = readFileSync(resolve(directory, file), 'utf8')
      for (const [, id] of source.matchAll(/\bid:\s*["']([^"']+)["']/g)) {
        expect(owners.has(id), `${id} in ${file} already used by ${owners.get(id)}`).toBe(false)
        owners.set(id, file)
      }
    }
  })
})
