/** Extract prose URLs without Markdown code-span delimiters or sentence punctuation. */
export function extractUrls(text) {
  return [...text.matchAll(/https?:\/\/[^\s)>\]"'`]+/g)].map((match) =>
    match[0].replace(/[.,;:!)\]]+$/, ''),
  )
}
