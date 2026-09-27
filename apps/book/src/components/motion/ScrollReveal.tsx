import { motion } from 'motion/react'
import type { ComponentChildren } from 'preact'
import { useEffect, useState } from 'preact/hooks'

export interface ScrollRevealProps {
  children: ComponentChildren
  delay?: number
  y?: number
  once?: boolean
  'data-testid'?: string
}

function useReducedMotionSafe(): boolean {
  const [reduce, setReduce] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches
  })
  useEffect(() => {
    if (typeof window === 'undefined') return
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const listener = (e: MediaQueryListEvent) => setReduce(e.matches)
    mq.addEventListener('change', listener)
    return () => mq.removeEventListener('change', listener)
  }, [])
  return reduce
}

export function ScrollReveal({
  children,
  delay = 0,
  y = 24,
  once = true,
  'data-testid': testId,
}: ScrollRevealProps) {
  const reduce = useReducedMotionSafe()
  if (reduce) {
    return (
      <div data-testid={testId} style={{ opacity: 1 }}>
        {children}
      </div>
    )
  }
  return (
    <motion.div
      data-testid={testId}
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once, amount: 0.25 }}
      transition={{ duration: 0.5, delay, ease: 'easeOut' }}
    >
      {children}
    </motion.div>
  )
}
