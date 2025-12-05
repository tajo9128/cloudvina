import { useEffect, useRef, useState } from 'react'

export default function MoleculeViewer({ pdbqtData, width = "100%", height = "100%", title = "3D Structure" }) {
    const viewerRef = useRef(null)
    const containerRef = useRef(null)
    const [isSpinning, setIsSpinning] = useState(false)

    useEffect(() => {
        if (!pdbqtData || !containerRef.current || !window.$3Dmol) return

        // Clear previous viewer
        if (viewerRef.current) {
            viewerRef.current.clear()
        }

        // Initialize 3Dmol viewer
        const viewer = window.$3Dmol.createViewer(containerRef.current, {
            backgroundColor: 'white'
        })

        // Add model from PDBQT data
        viewer.addModel(pdbqtData, 'pdbqt')

        // Set style - stick for bonds, sphere for atoms
        viewer.setStyle({}, {
            stick: { radius: 0.15 },
            sphere: { scale: 0.25 }
        })

        // Color by element
        viewer.setStyle({}, {
            stick: { colorscheme: 'Jmol' },
            sphere: { colorscheme: 'Jmol', scale: 0.25 }
        })

        // Center and zoom
        viewer.zoomTo()
        viewer.render()

        viewerRef.current = viewer

        return () => {
            if (viewerRef.current) {
                viewerRef.current.clear()
            }
        }
    }, [pdbqtData])

    const handleReset = () => {
        if (viewerRef.current) {
            viewerRef.current.zoomTo()
            viewerRef.current.render()
        }
    }

    const handleSpin = () => {
        if (viewerRef.current) {
            if (isSpinning) {
                viewerRef.current.spin(false)
            } else {
                viewerRef.current.spin('y', 1)
            }
            setIsSpinning(!isSpinning)
        }
    }

    const handleStyleChange = (style) => {
        if (!viewerRef.current) return

        switch (style) {
            case 'stick':
                viewerRef.current.setStyle({}, {
                    stick: { radius: 0.15, colorscheme: 'Jmol' }
                })
                break
            case 'sphere':
                viewerRef.current.setStyle({}, {
                    sphere: { scale: 0.4, colorscheme: 'Jmol' }
                })
                break
            case 'cartoon':
                viewerRef.current.setStyle({}, {
                    cartoon: { color: 'spectrum' }
                })
                break
            case 'both':
                viewerRef.current.setStyle({}, {
                    stick: { radius: 0.15, colorscheme: 'Jmol' },
                    sphere: { scale: 0.25, colorscheme: 'Jmol' }
                })
                break
        }
        viewerRef.current.render()
    }

    return (
        <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm h-full flex flex-col">
            {title && (
                <div className="mb-3 flex justify-between items-center">
                    <h3 className="font-semibold text-gray-900 text-lg">{title}</h3>
                    <div className="flex gap-2">
                        <button
                            onClick={handleReset}
                            className="text-sm px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded transition"
                            title="Reset view"
                        >
                            🔄 Reset
                        </button>
                        <button
                            onClick={handleSpin}
                            className={`text-sm px-3 py-1 rounded transition ${isSpinning
                                    ? 'bg-purple-100 text-purple-700 hover:bg-purple-200'
                                    : 'bg-gray-100 hover:bg-gray-200'
                                }`}
                            title="Toggle rotation"
                        >
                            {isSpinning ? '⏸ Stop' : '▶ Spin'}
                        </button>
                    </div>
                </div>
            )}

            {!title && (
                 <div className="mb-3 flex justify-end gap-2">
                    <button
                        onClick={handleReset}
                        className="text-sm px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded transition"
                        title="Reset view"
                    >
                        🔄 Reset
                    </button>
                    <button
                        onClick={handleSpin}
                        className={`text-sm px-3 py-1 rounded transition ${isSpinning
                                ? 'bg-purple-100 text-purple-700 hover:bg-purple-200'
                                : 'bg-gray-100 hover:bg-gray-200'
                            }`}
                        title="Toggle rotation"
                    >
                        {isSpinning ? '⏸ Stop' : '▶ Spin'}
                    </button>
                </div>
            )}

            {/* Style buttons */}
            <div className="mb-3 flex gap-2">
                <button
                    onClick={() => handleStyleChange('stick')}
                    className="text-xs px-2 py-1 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded"
                >
                    Stick
                </button>
                <button
                    onClick={() => handleStyleChange('sphere')}
                    className="text-xs px-2 py-1 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded"
                >
                    Sphere
                </button>
                <button
                    onClick={() => handleStyleChange('both')}
                    className="text-xs px-2 py-1 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded"
                >
                    Ball & Stick
                </button>
                <button
                    onClick={() => handleStyleChange('cartoon')}
                    className="text-xs px-2 py-1 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded"
                >
                    Cartoon
                </button>
            </div>

            {/* 3D Viewer Container */}
            <div
                ref={containerRef}
                style={{ width: width, height: height, minHeight: '400px' }}
                className="border border-gray-300 rounded bg-white flex-grow relative"
            />

            <div className="mt-2 text-xs text-gray-500">
                💡 <strong>Tip:</strong> Click and drag to rotate, scroll to zoom, right-click to pan
            </div>
        </div>
    )
}
