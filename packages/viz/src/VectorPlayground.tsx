import { svgColors, clamp } from './shared'

export interface Vector {
  x: number
  y: number
  label?: string
  color?: string
}

export interface VectorPlaygroundProps {
  vectors: Vector[]
  width?: number
  height?: number
  scale?: number
}

export function VectorPlayground({
  vectors,
  width = 320,
  height = 320,
  scale = 40,
}: VectorPlaygroundProps) {
  const c = svgColors()
  const cx = width / 2
  const cy = height / 2
  const maxCoord = Math.max(1, ...vectors.flatMap((v) => [Math.abs(v.x), Math.abs(v.y)]))
  const s = clamp(scale, 10, Math.min(width, height) / (maxCoord * 2 + 1))
  return (
    <svg role="img" aria-label="Vector playground" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <line x1={0} y1={cy} x2={width} y2={cy} stroke={c.muted} strokeWidth={1} />
      <line x1={cx} y1={0} x2={cx} y2={height} stroke={c.muted} strokeWidth={1} />
      {vectors.map((v, i) => {
        const x = cx + v.x * s
        const y = cy - v.y * s
        const color = v.color ?? c.accent
        return (
          <g key={i}>
            <line x1={cx} y1={cy} x2={x} y2={y} stroke={color} strokeWidth={2} markerEnd="url(#arrow)" />
            {v.label && (
              <text x={x + 6} y={y - 6} fill={c.fg} fontSize={12} fontFamily="var(--font-mono)">
                {v.label}
              </text>
            )}
          </g>
        )
      })}
      <defs>
        <marker id="arrow" markerWidth={8} markerHeight={8} refX={7} refY={4} orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill={c.accent} />
        </marker>
      </defs>
    </svg>
  )
}
