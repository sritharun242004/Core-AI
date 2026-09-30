import { describe, expect, it } from 'vitest'
import { WEEK_COLUMNS, compareWeeks, lessonsForColumn, weekLabel } from '../src/lib/weeks'

describe('split-week navigation', () => {
  it('distinguishes all specialization tracks without overwriting equal week numbers', () => {
    const entries = ['r', 'l', 'p'].map((track) => ({ week: 22, slug: `week-22${track}-lab` }))
    expect(entries.sort(compareWeeks).map(weekLabel)).toEqual(['22L', '22P', '22R'])
    const column = WEEK_COLUMNS.find((entry) => entry.week === 22)
    if (!column) throw new Error('Week 22 column missing')
    expect(lessonsForColumn(column, entries)).toHaveLength(3)
  })

  it('keeps separate 15a and 15b columns exact', () => {
    const entries = [
      { week: 15, slug: 'week-15a-sft-lora-dpo-lab' },
      { week: 15, slug: 'week-15b-moe-and-reasoning' },
    ]
    expect(lessonsForColumn(WEEK_COLUMNS[14], entries)).toEqual([entries[0]])
    expect(lessonsForColumn(WEEK_COLUMNS[15], entries)).toEqual([entries[1]])
  })

  it('labels the subweeks without inventing Week 16', () => {
    expect(weekLabel({ week: 15, slug: 'week-15a-sft-lora-dpo-lab' })).toBe('15a')
    expect(weekLabel({ week: 15, slug: 'week-15b-moe-and-reasoning' })).toBe('15b')
    expect(weekLabel({ week: 4, slug: 'week-04-python-info-theory' })).toBe('04')
    expect(WEEK_COLUMNS).toHaveLength(25)
    expect(WEEK_COLUMNS.map((c) => c.label).slice(13, 17)).toEqual(['14', '15a', '15b', '17'])
    expect(WEEK_COLUMNS.some((c) => c.week === 16)).toBe(false)
  })

  it('sorts 15a before 15b regardless of collection insertion order', () => {
    const entries = [
      { week: 17, slug: 'week-17-mini-rag-multimodal' },
      { week: 15, slug: 'week-15b-moe-and-reasoning' },
      { week: 14, slug: 'week-14-mini-bpe-pretrain' },
      { week: 15, slug: 'week-15a-sft-lora-dpo-lab' },
    ]
    expect(entries.sort(compareWeeks).map(weekLabel)).toEqual(['14', '15a', '15b', '17'])
  })
})
