import { useEffect, useRef, useState } from 'react'
import * as $3Dmol from '3dmol/build/3Dmol.js'

export default function MoleculeViewer({
    pdbqtData,
    receptorData,
    width = "100%",
    height = "100%",
    title = "3D Structure",
    interactions = null,
    cavities = null
}) {
    const viewerRef = useRef(null)
    const containerRef = useRef(null)
    const [isSpinning, setIsSpinning] = useState(false)
    const [showHBonds, setShowHBonds] = useState(true)
    const [showHydrophobic, setShowHydrophobic] = useState(true)
    const [showCavities, setShowCavities] = useState(true)
    const [showLabels, setShowLabels] = useState(false)
    const [currentStyle, setCurrentStyle] = useState('standard')

    useEffect(() => {
        if (!pdbqtData || !containerRef.current) return

        if (viewerRef.current) {
            viewerRef.current.clear()
        }

        const viewer = $3Dmol.createViewer(containerRef.current, {
            backgroundColor: 'white'
        })

        // Helper to check if string looks like PDBQT/PDB
        const isValidPDB = (str) => str && (str.includes('ATOM') || str.includes('HETATM') || str.includes('REMARK'))

        // Add Receptor (Protein)
        if (receptorData && isValidPDB(receptorData)) {
            viewer.addModel(receptorData, 'pdbqt')
        }

        // Add Ligand
        if (pdbqtData && isValidPDB(pdbqtData)) {
            viewer.addModel(pdbqtData, 'pdbqt')
        }

        // Apply style based on state
        viewer.removeAllSurfaces()
        switch (currentStyle) {
            case 'greenPink': // Publication
                viewer.setStyle({ hetflag: false }, {
                    cartoon: { color: '#22c55e', opacity: 1.0 }
                })
                viewer.setStyle({ hetflag: true }, {
                    stick: { color: '#db2777', radius: 0.25 }
                })
                break
            case 'surface': // Focused Surface
                // Protein as faint cartoon
                viewer.setStyle({ hetflag: false }, {
                    cartoon: { color: 'spectrum', opacity: 0.4 }
                })
                // Ligand as pink stick
                viewer.setStyle({ hetflag: true }, {
                    stick: { color: '#db2777', radius: 0.25 }
                })
                // Add surface ONLY for binding pocket (residues within 6A of ligand)
                const ligandSel = { hetflag: true }
                // Select atoms within 6A of ligand AND are protein (hetflag: false)
                const pocketSel = {
                    and: [
                        { hetflag: false },
                        { within: { distance: 6, sel: ligandSel } }
                    ]
                }
                viewer.addSurface($3Dmol.SurfaceType.VDW, {
                    opacity: 0.85,
                    color: 'white',
                }, pocketSel, pocketSel)
                break
            case 'standard':
            default:
                // Standard: Cartoon protein, Green stick ligand
                viewer.setStyle({ hetflag: false }, {
                    cartoon: { color: 'spectrum', opacity: 0.8 }, // Kept 'spectrum' from robust fix
                    stick: { colorscheme: 'chainHetatm', radius: 0.15, hidden: true }
                })
                viewer.setStyle({ hetflag: true }, {
                    stick: { colorscheme: 'greenCarbon', radius: 0.25 },
                    sphere: { colorscheme: 'greenCarbon', scale: 0.3 }
                })
                break
        }
    }, [pdbqtData, receptorData, interactions, cavities, showHBonds, showHydrophobic, showCavities, showLabels, currentStyle])

    useEffect(() => {
        const handleResize = () => { if (viewerRef.current) viewerRef.current.resize() }
        window.addEventListener('resize', handleResize)
        return () => window.removeEventListener('resize', handleResize)
    }, [])

    const handleReset = () => { if (viewerRef.current) { viewerRef.current.zoomTo(); viewerRef.current.render() } }
    const handleSpin = () => {
        if (viewerRef.current) {
            viewerRef.current.spin(isSpinning ? false : 'y')
            setIsSpinning(!isSpinning)
        }
    }
    const handleDownloadImage = () => {
        if (viewerRef.current) {
            const link = document.createElement('a')
            link.href = viewerRef.current.pngURI()
            link.download = `docking_${Date.now()}.png`
            link.click()
        }
    }
    const handleStyleChange = (style) => {
        if (!viewerRef.current) return
        const viewer = viewerRef.current
        viewer.setStyle({}, {}) // clear

        if (style === 'stick') {
            viewer.setStyle({ not: { hetflag: true } }, { stick: { radius: 0.15, colorscheme: 'chainHetatm' } })
            viewer.setStyle({ hetflag: true }, { stick: { radius: 0.25, colorscheme: 'greenCarbon' } })
        } else if (style === 'sphere') {
            viewer.setStyle({ hetflag: true }, { sphere: { scale: 0.4, colorscheme: 'greenCarbon' } })
        } else if (style === 'cartoon') {
            viewer.setStyle({ not: { hetflag: true } }, { cartoon: { color: 'spectrum', opacity: 0.8 } })
        } else {
            // Default: both style (cartoon protein + stick ligand)
            viewer.setStyle({ not: { hetflag: true } }, { cartoon: { color: 'spectrum', opacity: 0.8 } })
            viewer.setStyle({ hetflag: true }, { stick: { radius: 0.25, colorscheme: 'greenCarbon' } })
        }

        viewer.render()
        setCurrentStyle(style)
    }

    const hasInteractions = interactions && ((interactions.hydrogen_bonds?.length > 0) || (interactions.hydrophobic_contacts?.length > 0))
    const hasCavities = cavities && cavities.length > 0

    return (
        <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm h-full flex flex-col">
            <div className="mb-3 flex justify-between items-center flex-wrap gap-2">
                {title && <h3 className="font-semibold text-gray-900 text-lg">{title}</h3>}
                <div className="flex gap-2">
                    <button onClick={handleDownloadImage} className="text-sm px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded">📷 Snapshot</button>
                    <button onClick={handleReset} className="text-sm px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded">🔄 Reset</button>
                    <button onClick={handleSpin} className={`text-sm px-3 py-1 rounded ${isSpinning ? 'bg-purple-100 text-purple-700' : 'bg-gray-100 hover:bg-gray-200'}`}>{isSpinning ? '⏸ Stop' : '▶ Spin'}</button>
                </div>
            </div>

            <div className="mb-3 flex items-center justify-between flex-wrap gap-2">
                <div className="flex gap-2">
                    {[
                        { id: 'standard', label: '🌈 Standard' },
                        { id: 'greenPink', label: '🌿 Publication' },
                        { id: 'surface', label: '🧊 Surface' }
                    ].map(s => (
                        <button
                            key={s.id}
                            onClick={() => handleStyleChange(s.id)}
                            className={`text-xs px-3 py-1.5 rounded font-medium transition-colors ${currentStyle === s.id
                                ? 'bg-primary-100 text-primary-700 border border-primary-200'
                                : 'bg-slate-50 text-slate-600 border border-slate-200 hover:bg-slate-100'}`}
                        >
                            {s.label}
                        </button>
                    ))}
                </div>

                {hasInteractions && (
                    <div className="flex items-center gap-2">
                        <label className="inline-flex items-center cursor-pointer">
                            <input
                                type="checkbox"
                                checked={showLabels}
                                onChange={(e) => setShowLabels(e.target.checked)}
                                className="sr-only peer"
                            />
                            <div className="relative w-9 h-5 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600"></div>
                            <span className="ms-2 text-xs font-medium text-gray-700">Labels</span>
                        </label>
                    </div>
                )}
            </div>

            {(hasInteractions || hasCavities) && (
                <div className="mb-3 flex gap-3 text-sm">
                    {hasInteractions && (
                        <>
                            <label className="flex items-center gap-1 cursor-pointer">
                                <input type="checkbox" checked={showHBonds} onChange={e => setShowHBonds(e.target.checked)} className="rounded" />
                                <span className="text-blue-600">H-Bonds</span>
                            </label>
                            <label className="flex items-center gap-1 cursor-pointer">
                                <input type="checkbox" checked={showHydrophobic} onChange={e => setShowHydrophobic(e.target.checked)} className="rounded" />
                                <span className="text-yellow-600">Hydrophobic</span>
                            </label>
                        </>
                    )}
                    {hasCavities && (
                        <label className="flex items-center gap-1 cursor-pointer">
                            <input type="checkbox" checked={showCavities} onChange={e => setShowCavities(e.target.checked)} className="rounded" />
                            <span className="text-green-600">Cavities</span>
                        </label>
                    )}
                </div>
            )}

            <div ref={containerRef} style={{ width, height, minHeight: '400px', position: 'relative' }} className="border border-gray-300 rounded bg-white flex-grow" />

            {(hasInteractions || hasCavities) && (
                <div className="mt-2 flex gap-4 text-xs text-gray-600">
                    {showHBonds && interactions?.hydrogen_bonds?.length > 0 && <span>🔵 H-Bonds ({interactions.hydrogen_bonds.length})</span>}
                    {showHydrophobic && interactions?.hydrophobic_contacts?.length > 0 && <span>🟡 Hydrophobic ({interactions.hydrophobic_contacts.length})</span>}
                    {showCavities && hasCavities && <span>🟢 Cavities ({cavities.length})</span>}
                </div>
            )}

            <div className="mt-2 text-xs text-gray-500">💡 <strong>Tip:</strong> Drag to rotate, scroll to zoom, right-click to pan</div>
        </div>
    )
}
