# Architecture source and accessible fallback

Canonical source in the book's public directory:
`apps/book/public/diagrams/week-18-fsdp-ring.d2`.

The companion `week-18-fsdp-ring.svg` is a **hand-authored accessible fallback**, not
output falsely attributed to D2. The D2 CLI was absent in the implementation environment.
Both represent FSDP state materialization/reduction and a three-rank K/V ring. The SVG
has title/description, readable text, non-color-only labels, and the lesson adds alt
text, a caption, and prose equivalents. No animation or client-side renderer is required.

If D2 is already installed, from repository root generate a separate candidate:

```bash
d2 apps/book/public/diagrams/week-18-fsdp-ring.d2 \
   /tmp/week-18-fsdp-ring.generated.svg
```

Inspect the generated diagram before replacing the fallback. Preserve accessible SVG
title/description and check text wrapping, arrows, light/dark lesson contrast, and
small-screen readability. Generation was **not executed or validated** without the CLI.
Do not install tools, contact rendering services, or download diagram dependencies as
part of the offline notebook or test suite.
