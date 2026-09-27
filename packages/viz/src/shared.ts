export interface SvgColors {
  fg: string
  bg: string
  muted: string
  accent: string
}

export function svgColors(accent = 'var(--accent-p1)'): SvgColors {
  return {
    fg: 'var(--fg)',
    bg: 'var(--canvas)',
    muted: 'var(--fg-muted)',
    accent,
  }
}

export function clamp(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n))
}
