import { useEffect, useRef } from 'react'

// Sample caffeine molecule in SDF format for demo
const CAFFEINE_SDF = `
     RDKit          3D

 14 15  0  0  0  0  0  0  0  0999 V2000
    1.2124   -0.6957    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.3987   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.3987    1.3913    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.2124    2.0870    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.3913    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124   -0.6957    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124   -2.0870    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -2.7826    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.2124   -2.0870    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124    2.0870    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.6111   -0.6957    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.6111    2.0870    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4249   -2.7826    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  1  0
  4  5  2  0
  5  6  1  0
  6  1  1  0
  6  7  1  0
  7  8  1  0
  8  9  2  0
  9 10  1  0
 10  1  1  0
  5 11  1  0
  2 12  2  0
  3 13  1  0
  8 14  1  0
M  END
`

export default function Hero3DMol() {
    const containerRef = useRef(null)
    const viewerRef = useRef(null)

    useEffect(() => {
        if (!containerRef.current || !window.$3Dmol) return

        // Create viewer
        const viewer = window.$3Dmol.createViewer(containerRef.current, {
            backgroundColor: 'transparent',
            antialias: true
        })

        // Add caffeine molecule
        viewer.addModel(CAFFEINE_SDF, 'sdf')
        
        // Style with colorful atoms
        viewer.setStyle({}, {
            stick: { 
                radius: 0.2,
                colorscheme: 'Jmol'
            },
            sphere: { 
                scale: 0.3,
                colorscheme: 'Jmol'
            }
        })

        // Center and zoom
        viewer.zoomTo()
        viewer.zoom(1.5)
        
        // Start spinning
        viewer.spin('y', 0.5)
        
        viewer.render()
        viewerRef.current = viewer

        return () => {
            if (viewerRef.current) {
                viewerRef.current.spin(false)
                viewerRef.current.clear()
            }
        }
    }, [])

    return (
        <div 
            ref={containerRef}
            className="absolute right-0 top-1/2 -translate-y-1/2 w-[400px] h-[400px] lg:w-[500px] lg:h-[500px] opacity-30 lg:opacity-50 pointer-events-none"
            style={{ position: 'absolute' }}
        />
    )
}
