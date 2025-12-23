import { useEffect, useRef, useState } from 'react'
import * as $3Dmol from '3dmol/build/3Dmol.js'
import { Eye, RotateCw, Layers, Type } from 'lucide-react'

export default function MoleculeViewer({
    pdbqtData,
    receptorData,
    width = "100%",
    height = "100%",
    title = "3D Structure",
    interactions = null,
    cavities = null,
    bindingAffinity = null,
    receptorType = 'pdbqt',
    ligandType = 'pdbqt'
}) {
    const viewerRef = useRef(null)
    const containerRef = useRef(null)
    const [isSpinning, setIsSpinning] = useState(false)
    const [showHBonds, setShowHBonds] = useState(true)
    const [showHydrophobic, setShowHydrophobic] = useState(false)
    const [showCavities, setShowCavities] = useState(false)
    const [showLabels, setShowLabels] = useState(false)
    const [currentStyle, setCurrentStyle] = useState('standard') // 'standard', 'greenPink', 'cartoon', 'stick'
    const [showSurface, setShowSurface] = useState(false)

    // 1. Initialize Viewer ONCE
    useEffect(() => {
        if (!containerRef.current) return

        // Prevent duplicate initialization
        if (viewerRef.current) return

        const viewer = $3Dmol.createViewer(containerRef.current, {
            backgroundColor: 'white'
        })
        viewerRef.current = viewer

        return () => {
            // Cleanup if needed (3Dmol doesn't have a strict destroy, but we clear ref)
            viewerRef.current = null
        }
    }, [])

    // 2. Update Models & Styles
    useEffect(() => {
        const viewer = viewerRef.current
        if (!viewer) return

        // Clear previous state
        viewer.clear()

        // Helper to check if string looks like PDBQT/PDB
        const isValidPDB = (str) => str && (str.includes('ATOM') || str.includes('HETATM') || str.includes('REMARK'))

        // Add Receptor (Protein)
        if (receptorData && isValidPDB(receptorData)) {
            viewer.addModel(receptorData, receptorType)
            // Style receptor
            if (currentStyle === 'greenPink') {
                viewer.setStyle({ model: 0 }, { cartoon: { color: '#84cc16' } })
            } else if (currentStyle === 'cartoon' || currentStyle === 'standard') {
                viewer.setStyle({ model: 0 }, { cartoon: { color: 'spectrum' } })
            } else if (currentStyle === 'stick') {
                viewer.setStyle({ model: 0 }, { stick: { radius: 0.15, colorscheme: 'Jmol', opacity: 0.4 } })
            }
        }

        // Add Ligand
        if (pdbqtData && isValidPDB(pdbqtData)) {
            viewer.addModel(pdbqtData, ligandType)
            // Style ligand (always stick/sphere mix for visibility)
            if (currentStyle === 'greenPink') {
                viewer.setStyle({ model: -1 }, { stick: { colorscheme: 'magentaCarbon', radius: 0.25 } })
            } else {
                viewer.setStyle({ model: -1 }, { stick: { colorscheme: 'greenCarbon', radius: 0.25 } })
            }
        }

        // --- SURFACE ---
        if (showSurface) {
            // Surface is expensive, handle carefully
            try {
                viewer.addSurface($3Dmol.SurfaceType.VDW, {
                    opacity: 0.7,
                    color: 'white'
                }, { hetflag: false })
            } catch (e) { console.warn("Surface generation error", e) }
        }

        // --- LABELS ---
        if (showLabels) {
            viewer.addPropertyLabels("resn", { hetflag: false }, {
                fontColor: 'black', font: 'sans-serif', fontSize: 10, showBackground: false, alignment: 'center'
            })
            viewer.addPropertyLabels("resn", { hetflag: true }, {
                fontColor: 'red', font: 'sans-serif', fontSize: 12, showBackground: true, backgroundColor: 'white'
            })
        }

        // --- INTERACTIONS ---
        if (showHBonds && interactions?.hydrogen_bonds) {
            interactions.hydrogen_bonds.forEach(bond => {
                let resNum = null;
                const resStr = bond.receptor_residue || "";
                const match = resStr.match(/(\d+)/);
                if (match) resNum = parseInt(match[1]);

                let pAtoms = viewer.selectedAtoms({ resi: resNum, atom: bond.protein_atom });
                if (pAtoms.length === 0 && resNum) pAtoms = viewer.selectedAtoms({ resi: resNum });

                let lAtoms = viewer.selectedAtoms({ hetflag: true, atom: bond.ligand_atom });

                if (pAtoms.length > 0 && lAtoms.length > 0) {
                    viewer.addCylinder({
                        start: { x: pAtoms[0].x, y: pAtoms[0].y, z: pAtoms[0].z },
                        end: { x: lAtoms[0].x, y: lAtoms[0].y, z: lAtoms[0].z },
                        radius: 0.15, color: 'yellow', dashed: true
                    });
                }
            })
        }

        viewer.zoomTo();
        viewer.render();

    }, [pdbqtData, receptorData, currentStyle, showSurface, showLabels, showHBonds, interactions])

    // Spin Effect
    useEffect(() => {
        if (!viewerRef.current) return;
        if (isSpinning) {
            viewerRef.current.spin("y", 1);
        } else {
            viewerRef.current.spin(false);
        }
    }, [isSpinning]);


    return (
        <div className="relative w-full h-full bg-white rounded-xl overflow-hidden border border-slate-200 shadow-sm group">
            <div ref={containerRef} className="w-full h-full cursor-move" />

            {/* Title Overlay */}
            <div className="absolute top-4 left-4 z-10 pointer-events-none">
                <h3 className="text-sm font-bold text-slate-700 bg-white/80 backdrop-blur px-3 py-1.5 rounded-lg shadow-sm">
                    {title || "3D Visualization"}
                    {/* show simple affinity if provided */}
                    {bindingAffinity && <span className="ml-2 text-indigo-600">{bindingAffinity.toFixed(1)} kcal/mol</span>}
                </h3>
            </div>

            {/* Controls Bar (Bottom) */}
            <div className="absolute bottom-4 left-4 right-4 z-10 flex justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none">
                <div className="flex items-center gap-2 bg-slate-900/80 backdrop-blur text-white px-2 py-2 rounded-xl shadow-lg pointer-events-auto">

                    {/* Spin */}
                    <button
                        onClick={() => setIsSpinning(!isSpinning)}
                        className={`p-2 rounded-lg transition-colors ${isSpinning ? 'bg-indigo-500 text-white' : 'hover:bg-white/10 text-slate-300'}`}
                        title="Toggle Spin"
                    >
                        <RotateCw size={18} className={isSpinning ? 'animate-spin' : ''} />
                    </button>

                    <div className="w-px h-6 bg-white/20 mx-1"></div>

                    {/* Styles */}
                    <button
                        onClick={() => setCurrentStyle('cartoon')}
                        className={`px-3 py-1.5 text-xs font-bold rounded-md transition-colors ${currentStyle === 'cartoon' || currentStyle === 'standard' ? 'bg-white/20 text-white' : 'hover:bg-white/10 text-slate-300'}`}
                    >
                        Cartoon
                    </button>
                    <button
                        onClick={() => setCurrentStyle('stick')}
                        className={`px-3 py-1.5 text-xs font-bold rounded-md transition-colors ${currentStyle === 'stick' ? 'bg-white/20 text-white' : 'hover:bg-white/10 text-slate-300'}`}
                    >
                        Stick
                    </button>

                    <div className="w-px h-6 bg-white/20 mx-1"></div>

                    {/* Toggles */}
                    <button
                        onClick={() => setShowSurface(!showSurface)}
                        className={`p-2 rounded-lg transition-colors ${showSurface ? 'bg-indigo-500 text-white' : 'hover:bg-white/10 text-slate-300'}`}
                        title="Toggle Molecular Surface"
                    >
                        <Layers size={18} />
                    </button>

                    <button
                        onClick={() => setShowLabels(!showLabels)}
                        className={`p-2 rounded-lg transition-colors ${showLabels ? 'bg-indigo-500 text-white' : 'hover:bg-white/10 text-slate-300'}`}
                        title="Toggle Labels"
                    >
                        <Type size={18} />
                    </button>

                    <button
                        onClick={() => setShowHBonds(!showHBonds)}
                        className={`p-2 rounded-lg transition-colors ${showHBonds ? 'bg-indigo-500 text-white' : 'hover:bg-white/10 text-slate-300'}`}
                        title="Toggle Interactions"
                    >
                        <Eye size={18} />
                    </button>
                </div>
            </div>
        </div>
    )
}
