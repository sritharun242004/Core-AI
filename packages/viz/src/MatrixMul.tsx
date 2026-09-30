import { svgColors } from './shared'

export interface MatrixMulProps {
  a: number[][]
  b: number[][]
  width?: number
  height?: number
  highlight?: { i: number; j: number }
}

export function MatrixMul({ a, b, width = 480, height = 200, highlight }: MatrixMulProps) {
  const c = svgColors()
  const cell = 32
  const gap = 24
  const rowsA = a.length
  const colsA = a[0]?.length ?? 0
  const colsB = b[0]?.length ?? 0

  const drawGrid = (m: number[][], ox: number, label: string, kind: 'a' | 'b') => (
    <g>
      <text x={ox} y={16} fill={c.muted} fontSize={12} fontFamily="var(--font-mono)">
        {label}
      </text>
      {m.map((row, i) =>
        row.map((val, j) => {
          const isHi =
            highlight &&
            ((kind === 'a' && i === highlight.i) || (kind === 'b' && j === highlight.j))
          return (
            // biome-ignore lint/suspicious/noArrayIndexKey: matrix coordinates are cell identity, even when their values change.
            <g key={`${kind}-${i}-${j}`} data-cell>
              <rect
                x={ox + j * cell}
                y={24 + i * cell}
                width={cell}
                height={cell}
                fill={isHi ? c.accent : 'transparent'}
                fillOpacity={isHi ? 0.15 : 0}
                stroke={c.muted}
              />
              <text
                x={ox + j * cell + cell / 2}
                y={24 + i * cell + cell / 2 + 4}
                textAnchor="middle"
                fill={c.fg}
                fontSize={12}
                fontFamily="var(--font-mono)"
              >
                {val}
              </text>
            </g>
          )
        }),
      )}
    </g>
  )

  const oxA = 8
  const oxB = oxA + colsA * cell + gap
  return (
    <svg
      role="img"
      aria-label="Matrix multiplication"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
    >
      {drawGrid(a, oxA, `A (${rowsA}×${colsA})`, 'a')}
      {drawGrid(b, oxB, `B (${b.length}×${colsB})`, 'b')}
    </svg>
  )
}
