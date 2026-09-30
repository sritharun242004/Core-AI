import { expect, it } from 'vitest'
import { extractUrls } from '../../../scripts/link-urls.mjs'

it('excludes inline-code backticks and sentence punctuation from URLs', () => {
  expect(extractUrls('Visit `http://localhost:4321` or https://example.com/docs.')).toEqual([
    'http://localhost:4321',
    'https://example.com/docs',
  ])
})

it('retains path, query and fragment in Markdown destinations', () => {
  expect(extractUrls('[paper](https://example.org/paper?v=2#math)')).toEqual([
    'https://example.org/paper?v=2#math',
  ])
})
