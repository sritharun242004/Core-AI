import { motion, useReducedMotion } from 'motion/react'
import type { ComponentChildren } from 'preact'

export interface ScrollRevealProps {
  children: ComponentChildren
  delay?: number
  y?: number
  once?: boolean
  'data-testid'?: string
}

export function ScrollReveal({
  children,
  delay = 0,
  y = 24,
  once = true,
  'data-testid': testId,
}: ScrollRevealProps) {
  const reduce = useReducedMotion()
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
