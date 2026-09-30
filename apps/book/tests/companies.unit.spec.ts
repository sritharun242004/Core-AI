import { describe, expect, it } from 'vitest'
import { COMPANY_SLUGS, assertAll7 } from '../src/lib/companies'

describe('company parity values', () => {
  it('rejects present-but-empty entries', () => {
    const record = Object.fromEntries(COMPANY_SLUGS.map((slug) => [slug, 'source']))
    expect(() => assertAll7({ ...record, qwen: undefined }, 'sources')).toThrow(/qwen/)
    expect(() => assertAll7({ ...record, qwen: null }, 'sources')).toThrow(/qwen/)
  })

  it('accepts all seven own values and rejects inherited values', () => {
    const record = Object.fromEntries(COMPANY_SLUGS.map((slug) => [slug, 'source']))
    expect(() => assertAll7(record, 'sources')).not.toThrow()
    expect(() => assertAll7(Object.create(record), 'sources')).toThrow(/missing/)
  })
})
