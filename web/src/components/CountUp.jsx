import { useState, useEffect, useRef } from 'react'

export default function CountUp({ end, duration = 2000, suffix = '' }) {
    const [count, setCount] = useState(0)
    const countRef = useRef(null)
    const [isVisible, setIsVisible] = useState(false)

    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setIsVisible(true)
                    observer.disconnect()
                }
            },
            { threshold: 0.1 }
        )

        if (countRef.current) {
            observer.observe(countRef.current)
        }

        return () => observer.disconnect()
    }, [])

    useEffect(() => {
        if (!isVisible) return

        let startTime
        let animationFrame

        const animate = (timestamp) => {
            if (!startTime) startTime = timestamp
            const progress = timestamp - startTime

            if (progress < duration) {
                const percentage = progress / duration
                // Ease out quart
                const ease = 1 - Math.pow(1 - percentage, 4)

                setCount(Math.floor(end * ease))
                animationFrame = requestAnimationFrame(animate)
            } else {
                setCount(end)
            }
        }

        animationFrame = requestAnimationFrame(animate)

        return () => cancelAnimationFrame(animationFrame)
    }, [isVisible, end, duration])

    return (
        <span ref={countRef} className="tabular-nums">
            {count.toLocaleString()}{suffix}
        </span>
    )
}
