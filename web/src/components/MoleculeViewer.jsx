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
    const [showHBonds, setShowHBonds] = useState(true)
    const [showHydrophobic, setShowHydrophobic] = useState(true)
    const [showCavities, setShowCavities] = useState(true)

    useEffect(() => {
        if (!pdbqtData || !containerRef.current || !window.$3Dmol) return

        if (viewerRef.current) {
            viewerRef.current.clear()
        }

        const viewer = window.$3Dmol.createViewer(containerRef.current, {
            backgroundColor: 'white'
        })

        const models = viewer.addModel(pdbqtData, 'pdbqt')

        // Style protein with cartoon (colorful spectrum) + sticks
        viewer.setStyle({ hetflag: false }, {
            cartoon: { color: 'spectrum', opacity: 0.8 },
            stick: { colorscheme: 'chainHetatm', radius: 0.15 }
        })

        // Style ligand with ball-and-stick (vibrant colors)
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

        // Cavity spheres (reduced size for less dominance)
        if (cavities && showCavities && cavities.length > 0) {
            const colors = ['#22c55e', '#f97316', '#a855f7', '#ec4899', '#14b8a6']
            cavities.forEach((cavity, i) => {
                viewer.addSphere({
                    center: { x: cavity.center_x, y: cavity.center_y, z: cavity.center_z },
                    radius: Math.min(cavity.size_x, cavity.size_y, cavity.size_z) / 8, // Reduced from /4 to /8
                    color: colors[i % colors.length], opacity: 0.15 // Reduced opacity from 0.3 to 0.15
                })
                viewer.addLabel(`P${cavity.pocket_id}`, { // Shortened label
                    position: { x: cavity.center_x, y: cavity.center_y + 2, z: cavity.center_z },
                    backgroundColor: colors[i % colors.length], fontColor: 'white', fontSize: 10, // Smaller font
                    backgroundOpacity: 0.7
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
        const styles = {
            stick: {
                hetflag: false, // protein
                stick: { radius: 0.15, colorscheme: 'chainHetatm' },
                hetflag: true, // ligand
                stick: { radius: 0.25, colorscheme: 'greenCarbon' }
            },
            sphere: {
                hetflag: true, // ligand only
                sphere: { scale: 0.4, colorscheme: 'greenCarbon' }
            },
            cartoon: {
                hetflag: false, // protein
                cartoon: { color: 'spectrum', opacity: 0.8 }
            },
        }

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
            viewer.setStyle({ not: { hetflag: true } }, { cartoon: { color: 'spectrum', opacity: 0.8 } })
            viewer.setStyle({ hetflag: true }, { stick: { radius: 0.25, colorscheme: 'greenCarbon' } })
        }

        viewer.render()
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
                {['stick', 'sphere', 'both', 'cartoon'].map(s => (
                    <button key={s} onClick={() => handleStyleChange(s)} className="text-xs px-2 py-1 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded capitalize">{s === 'both' ? 'Ball & Stick' : s}</button>
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
