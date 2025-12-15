import { useEffect, useRef, useState } from 'react'

export default function MoleculeViewer({
    pdbqtData,
    width = "100%",
    height = "100%",
    title = "3D Structure",
    interactions = null,
    cavities = null
}) {
    const viewerRef = useRef(null)
    const containerRef = useRef(null)
    const [isSpinning, setIsSpinning] = useState(false)
    const [isSpinning, setIsSpinning] = useState(false)
    const [showHBonds, setShowHBonds] = useState(true)
    const [showHydrophobic, setShowHydrophobic] = useState(true)
    const [showCavities, setShowCavities] = useState(true)
    const [currentStyle, setCurrentStyle] = useState('standard')

    useEffect(() => {
        if (!pdbqtData || !containerRef.current || !window.$3Dmol) return

        if (viewerRef.current) {
            viewerRef.current.clear()
        }

        const viewer = window.$3Dmol.createViewer(containerRef.current, {
            backgroundColor: 'white'
        })

        viewer.addModel(pdbqtData, 'pdbqt')

        // Style protein with cartoon (colorful spectrum)
        viewer.setStyle({ hetflag: false }, {
            cartoon: { color: 'spectrum', opacity: 0.8 },
            stick: { colorscheme: 'chainHetatm', radius: 0.15 }
        })

        // Style ligand with ball-and-stick (vibrant green)
        viewer.setStyle({ hetflag: true }, {
            stick: { colorscheme: 'greenCarbon', radius: 0.25 },
            sphere: { colorscheme: 'greenCarbon', scale: 0.3 }
        })

        // H-Bonds visualization (blue dashed lines)
        if (interactions && showHBonds && interactions.hydrogen_bonds) {
            interactions.hydrogen_bonds.forEach(bond => {
                if (bond.ligand_coords && bond.protein_coords) {
                    viewer.addCylinder({
                        start: { x: bond.ligand_coords[0], y: bond.ligand_coords[1], z: bond.ligand_coords[2] },
                        end: { x: bond.protein_coords[0], y: bond.protein_coords[1], z: bond.protein_coords[2] },
                        radius: 0.08, color: '#2563eb', dashed: true, dashLength: 0.25, gapLength: 0.1
                    })
                }
            })
        }

        // Hydrophobic contacts (yellow dotted lines)
        if (interactions && showHydrophobic && interactions.hydrophobic_contacts) {
            interactions.hydrophobic_contacts.forEach(contact => {
                if (contact.ligand_coords && contact.protein_coords) {
                    viewer.addCylinder({
                        start: { x: contact.ligand_coords[0], y: contact.ligand_coords[1], z: contact.ligand_coords[2] },
                        end: { x: contact.protein_coords[0], y: contact.protein_coords[1], z: contact.protein_coords[2] },
                        radius: 0.05, color: '#eab308', dashed: true, dashLength: 0.15, gapLength: 0.15
                    })
                }
            })
        }

        // Cavity spheres
        if (cavities && showCavities && cavities.length > 0) {
            const colors = ['#22c55e', '#f97316', '#a855f7', '#ec4899', '#14b8a6']
            cavities.forEach((cavity, i) => {
                viewer.addSphere({
                    center: { x: cavity.center_x, y: cavity.center_y, z: cavity.center_z },
                    radius: Math.min(cavity.size_x, cavity.size_y, cavity.size_z) / 4,
                    color: colors[i % colors.length], opacity: 0.3
                })
                viewer.addLabel(`Pocket ${cavity.pocket_id}`, {
                    position: { x: cavity.center_x, y: cavity.center_y + 3, z: cavity.center_z },
                    backgroundColor: colors[i % colors.length], fontColor: 'white', fontSize: 12
                })
            })
        }

        viewer.zoomTo()
        viewer.render()
        viewerRef.current = viewer

        return () => { if (viewerRef.current) viewerRef.current.clear() }
    }, [pdbqtData, interactions, cavities, showHBonds, showHydrophobic, showCavities])

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
        setCurrentStyle(style)

        switch (style) {
            case 'greenPink': // Publication: Green Protein + Pink Ligand
                viewerRef.current.setStyle({ hetflag: false }, {
                    cartoon: { color: '#22c55e', opacity: 1.0 } // Green protein
                })
                viewerRef.current.setStyle({ hetflag: true }, {
                    stick: { color: '#db2777', radius: 0.25 } // Pink ligand
                })
                break
            case 'surface': // Surface visualization
                viewerRef.current.setStyle({ hetflag: false }, {
                    surface: { opacity: 0.8, colorscheme: 'greenCarbon' } // Surface
                })
                viewerRef.current.setStyle({ hetflag: true }, {
                    stick: { color: 'white', radius: 0.25 }, // White ligand for contrast
                    sphere: { color: 'white', scale: 0.3 }
                })
                break
            case 'standard': // Default: Spectrum + Green Ligand
            default:
                viewerRef.current.setStyle({ hetflag: false }, {
                    cartoon: { color: 'spectrum', opacity: 0.8 },
                    stick: { colorscheme: 'chainHetatm', radius: 0.15 }
                })
                viewerRef.current.setStyle({ hetflag: true }, {
                    stick: { colorscheme: 'greenCarbon', radius: 0.25 },
                    sphere: { colorscheme: 'greenCarbon', scale: 0.3 }
                })
                break
        }
        viewerRef.current.render()
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

            <div className="mb-3 flex gap-2">
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
