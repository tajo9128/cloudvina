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

        // RECEPTOR (Protein) Style
        // Select logic: Not hetatms (usually protein) AND not water
        const proteinSel = { hetflag: false, invert: true } // Logic vary, let's stick to simple

        // LIGAND Style
        // Select logic: Hetatms (usually ligand in PDBQT)
        const ligandSel = { hetflag: true }

        // ... existing styles ...
        switch (currentStyle) {
            case 'greenPink': // PyMOL-like Publication Style (Green Protein, Magenta Ligand)
                viewer.setStyle({}, { cartoon: { color: '#84cc16' } })
                viewer.setStyle({ hetflag: true }, { stick: { colorscheme: 'magentaCarbon', radius: 0.25 } })

                // Show Interacting Residues as Sticks if interactions exist
                if (interactions) {
                    const residuesToShow = new Set();
                    // Collect residue numbers
                    [...(interactions.hydrogen_bonds || []), ...(interactions.hydrophobic_contacts || [])].forEach(i => {
                        const resInfo = i.receptor_residue || i.residue;
                        if (resInfo) {
                            // resInfo format often "TYR123" or "A:123" or similar. Need to parse.
                            // Assuming format "RESNUM" or "CHAIN:RESNUM" or number.
                            // The backend normally returns string "TYR123". Regex to get number.
                            const match = resInfo.match(/(\d+)/);
                            if (match) residuesToShow.add(parseInt(match[1]));
                        }
                    });

                    residuesToShow.forEach(resi => {
                        viewer.addStyle({ resi: resi }, { stick: { colorscheme: 'greenCarbon', radius: 0.2 } });
                    });
                }
                break;

            case 'surface': // Surface View
                viewer.setStyle({}, { cartoon: { color: 'spectrum', opacity: 0.5 } })
                viewer.addSurface($3Dmol.SurfaceType.MS, {
                    opacity: 0.85,
                    color: 'white'
                }, { hetflag: false })
                viewer.setStyle({ hetflag: true }, { stick: { colorscheme: 'redCarbon', radius: 0.3 } })
                break;

            case 'standard':
            default:
                viewer.setStyle({ hetflag: false }, { cartoon: { color: 'spectrum' } })
                viewer.setStyle({ hetflag: true }, { stick: { colorscheme: 'greenCarbon', radius: 0.25 } })
                break;
        }

        // --- CAVITY RENDERING ---
        if (showCavities && cavities) {
            cavities.forEach(pocket => {
                // Calculate approximate radius from volume (V = 4/3 * pi * r^3)
                // r = cube_root(V * 3 / (4 * pi))
                const radius = Math.pow((pocket.volume * 3) / (4 * Math.PI), 1 / 3); // Cube root

                // Add Cloud/Sphere represention
                viewer.addSphere({
                    center: { x: pocket.center_x, y: pocket.center_y, z: pocket.center_z },
                    radius: radius,
                    color: 'cyan',
                    alpha: 0.4,
                    wireframe: true
                });

                // Add visible label for the pocket
                viewer.addLabel(`Pocket ${pocket.pocket_id}`, {
                    position: { x: pocket.center_x, y: pocket.center_y, z: pocket.center_z },
                    backgroundColor: 'black',
                    fontColor: 'white',
                    fontSize: 12,
                    showBackground: true,
                    backgroundOpacity: 0.7
                });
            });
        }

        viewer.render()
        viewer.zoomTo()
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
