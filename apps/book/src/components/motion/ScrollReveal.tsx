import type { ComponentChildren } from 'preact'
import { useEffect, useRef, useState } from 'preact/hooks'

export interface ScrollRevealProps {
  children: ComponentChildren
  delay?: number
  y?: number
  once?: boolean
  'data-testid'?: string
}

/** Progressive enhancement: SSR/no-JS content is always readable. */
export function ScrollReveal({
  children,
  delay = 0,
  y = 24,
  once = true,
  'data-testid': testId,
}: ScrollRevealProps) {
  const element = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(true)
  const [reduced, setReduced] = useState(true)

  useEffect(() => {
    const node = element.current
    if (!node || typeof window.matchMedia !== 'function') return
    const query = window.matchMedia('(prefers-reduced-motion: reduce)')
    let observer: IntersectionObserver | undefined
    const configure = () => {
      observer?.disconnect()
      setReduced(query.matches)
      setVisible(true)
      if (query.matches || typeof IntersectionObserver === 'undefined') return
      // Only hide offscreen content after the browser can observe it.
      setVisible(node.getBoundingClientRect().top < window.innerHeight)
      observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            setVisible(true)
            if (once) observer?.disconnect()
          } else if (!once) setVisible(false)
        },
        { threshold: 0.1 },
      )
      observer.observe(node)
    }
    configure()
    query.addEventListener('change', configure)
    return () => {
      observer?.disconnect()
      query.removeEventListener('change', configure)
    }
  }, [once])

  return (
    <div
      ref={element}
      data-testid={testId}
      style={{
        opacity: visible ? 1 : 0,
        transform: visible || reduced ? 'none' : `translateY(${y}px)`,
        transition: reduced
          ? 'none'
          : 'opacity 240ms cubic-bezier(0.16,1,0.3,1), transform 240ms cubic-bezier(0.16,1,0.3,1)',
        transitionDelay: reduced ? '0s' : `${Math.max(0, delay)}s`,
      }}
    >
      {children}
    </div>
  )
}
