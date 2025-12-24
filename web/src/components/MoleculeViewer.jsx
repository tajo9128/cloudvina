import { useEffect, useRef, useState } from 'react'
import * as $3Dmol from '3dmol/build/3Dmol.js'
import { Eye, RotateCw, Layers, Type, Maximize, Focus, Palette, BookOpen } from 'lucide-react'

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
    const [showLabels, setShowLabels] = useState(false)
    const [showSurface, setShowSurface] = useState(false)
    const [isFullscreen, setIsFullscreen] = useState(false)

    // Presets: 'publication' (PyMOL Style), 'analysis' (Detailed), 'surface' (Pocket View)
    const [currentPreset, setCurrentPreset] = useState('publication')

    // 1. Initialize Viewer
    useEffect(() => {
        if (!containerRef.current) return
        if (viewerRef.current) return

        const viewer = $3Dmol.createViewer(containerRef.current, {
            backgroundColor: 'white'
        })
        viewerRef.current = viewer

        return () => { viewerRef.current = null }
    }, [])

    // 2. Render Scene based on Preset
    useEffect(() => {
        const viewer = viewerRef.current
        if (!viewer) return

        viewer.clear()
        const isValidPDB = (str) => str && (str.includes('ATOM') || str.includes('HETATM') || str.includes('REMARK'))

        // Preset Configurations
        const config = {
            publication: {
                receptor: { style: 'cartoon', color: 'white', opacity: 1.0 },
                ligand: { style: 'stick', color: 'greenCarbon', radius: 0.25 },
                surface: false,
                hbonds: true
            },
            analysis: {
                receptor: { style: 'cartoon', color: 'spectrum', opacity: 0.8 },
                ligand: { style: 'stick', color: 'cyanCarbon', radius: 0.25 },
                surface: false,
                hbonds: true
            },
            surface: {
                receptor: { style: 'cartoon', color: 'white', opacity: 1.0 },
                ligand: { style: 'stick', color: 'greenCarbon', radius: 0.25 },
                surface: true,
                hbonds: false
            }
        }[currentPreset]

        // --- ADD RECEPTOR ---
        if (receptorData && isValidPDB(receptorData)) {
            viewer.addModel(receptorData, receptorType)
            const style = {}
            if (config.receptor.style === 'cartoon') {
                style.cartoon = {
                    color: config.receptor.color === 'spectrum' ? 'spectrum' : config.receptor.color,
                    opacity: config.receptor.opacity
                }
            }
            viewer.setStyle({ model: 0 }, style)
        }

        // --- ADD LIGAND ---
        if (pdbqtData && isValidPDB(pdbqtData)) {
            viewer.addModel(pdbqtData, ligandType)
            viewer.setStyle({ model: -1 }, {
                stick: {
                    colorscheme: config.ligand.color,
                    radius: config.ligand.radius
                }
            })
        }

        // --- OPTIONAL: SURFACE ---
        if (showSurface || config.surface) {
            try {
                viewer.addSurface($3Dmol.SurfaceType.VDW, {
                    opacity: 0.85,
                    color: 'white',
                }, { hetflag: false })
            } catch (e) { console.warn("Surface error", e) }
        }

        // --- LABELS ---
        if (showLabels) {
            viewer.addPropertyLabels("resn", { hetflag: false }, {
                fontColor: 'black', font: 'sans-serif', fontSize: 10, showBackground: false, alignment: 'center'
            })
        }

        // --- INTERACTIONS (H-Bonds) ---
        if ((showHBonds || config.hbonds) && interactions?.hydrogen_bonds) {
            interactions.hydrogen_bonds.forEach(bond => {
                let resNum = null;
                const match = (bond.receptor_residue || "").match(/(\d+)/);
                if (match) resNum = parseInt(match[1]);

                let pAtoms = viewer.selectedAtoms({ resi: resNum, atom: bond.protein_atom });
                if (pAtoms.length === 0 && resNum) pAtoms = viewer.selectedAtoms({ resi: resNum });
                let lAtoms = viewer.selectedAtoms({ hetflag: true, atom: bond.ligand_atom });

                if (pAtoms.length > 0 && lAtoms.length > 0) {
                    // PyMOL Style: Yellow Dashes
                    viewer.addCylinder({
                        start: { x: pAtoms[0].x, y: pAtoms[0].y, z: pAtoms[0].z },
                        end: { x: lAtoms[0].x, y: lAtoms[0].y, z: lAtoms[0].z },
                        radius: 0.1, color: 'yellow', dashed: true, dashLength: 0.2
                    });

                    // Label distance
                    if (showLabels) {
                        viewer.addLabel(`${bond.distance}Å`, {
                            position: {
                                x: (pAtoms[0].x + lAtoms[0].x) / 2,
                                y: (pAtoms[0].y + lAtoms[0].y) / 2,
                                z: (pAtoms[0].z + lAtoms[0].z) / 2
                            },
                            backgroundColor: 'black', fontColor: 'white', fontSize: 10, showBackground: true
                        })
                    }
                }
            })
        }

        viewer.zoomTo({ hetflag: true }, 2000); // Focus on Ligand with animation
        viewer.render();

    }, [pdbqtData, receptorData, currentPreset, showSurface, showLabels, showHBonds, interactions])

    // Spin Effect
    useEffect(() => {
        if (!viewerRef.current) return;
        viewerRef.current.spin("y", isSpinning ? 0.5 : 0);
    }, [isSpinning]);

    const handleFullscreen = () => {
        if (!containerRef.current) return;
        if (!isFullscreen) {
            if (containerRef.current.requestFullscreen) containerRef.current.requestFullscreen();
            setIsFullscreen(true);
        } else {
            if (document.exitFullscreen) document.exitFullscreen();
            setIsFullscreen(false);
        }
    }

    return (
        <div className={`relative w-full h-full bg-white rounded-xl overflow-hidden border border-slate-200 shadow-sm group ${isFullscreen ? 'fixed inset-0 z-50 rounded-none' : ''}`}>

            <div ref={containerRef} className="w-full h-full cursor-move" />

            {/* Top Bar: Title & Key Stats */}
            <div className="absolute top-4 left-4 z-10 pointer-events-none flex flex-col gap-2">
                <h3 className="text-sm font-bold text-slate-700 bg-white/90 backdrop-blur px-3 py-1.5 rounded-lg shadow-sm w-fit border border-slate-200">
                    {title}
                </h3>
                {bindingAffinity && (
                    <div className="bg-indigo-600/90 backdrop-blur text-white px-3 py-1.5 rounded-lg shadow-sm text-xs font-bold w-fit">
                        Affinity: {bindingAffinity.toFixed(1)} kcal/mol
                    </div>
                )}
            </div>

            {/* Bottom Floating Toolbar */}
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center gap-2 opacity-0 group-hover:opacity-100 transition-all duration-300">

                {/* Presets Row */}
                <div className="flex bg-white/90 backdrop-blur rounded-xl p-1 shadow-lg border border-slate-200">
                    <button onClick={() => setCurrentPreset('publication')} className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-2 transition-all ${currentPreset === 'publication' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:bg-slate-100'}`}>
                        <BookOpen size={14} /> PyMOL View
                    </button>
                    <button onClick={() => setCurrentPreset('analysis')} className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-2 transition-all ${currentPreset === 'analysis' ? 'bg-blue-600 text-white' : 'text-slate-500 hover:bg-slate-100'}`}>
                        <Palette size={14} /> Analysis
                    </button>
                    <button onClick={() => setCurrentPreset('surface')} className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-2 transition-all ${currentPreset === 'surface' ? 'bg-emerald-600 text-white' : 'text-slate-500 hover:bg-slate-100'}`}>
                        <Layers size={14} /> Surface
                    </button>
                </div>

                {/* Controls Row */}
                <div className="flex items-center gap-1 bg-slate-900/90 backdrop-blur text-white p-1.5 rounded-xl shadow-xl border border-white/10">
                    <button onClick={() => isSpinning ? setIsSpinning(false) : setIsSpinning(true)} className={`p-2 rounded-lg transition-colors ${isSpinning ? 'bg-indigo-500 text-white' : 'hover:bg-white/10 text-slate-400'}`} title="Auto-Rotate">
                        <RotateCw size={18} className={isSpinning ? 'animate-spin' : ''} />
                    </button>

                    <div className="w-px h-6 bg-white/20 mx-1"></div>

                    <button onClick={() => { viewerRef.current?.zoomTo({ hetflag: true }, 1000) }} className="p-2 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors" title="Focus Ligand">
                        <Focus size={18} />
                    </button>

                    <button onClick={() => setShowLabels(!showLabels)} className={`p-2 rounded-lg transition-colors ${showLabels ? 'bg-indigo-500 text-white' : 'hover:bg-white/10 text-slate-400'}`} title="Toggle Labels">
                        <Type size={18} />
                    </button>

                    <button onClick={() => setShowHBonds(!showHBonds)} className={`p-2 rounded-lg transition-colors ${showHBonds ? 'bg-yellow-500/20 text-yellow-400' : 'hover:bg-white/10 text-slate-400'}`} title="Toggle H-Bonds">
                        <span className="font-bold text-xs">HB</span>
                    </button>

                    <div className="w-px h-6 bg-white/20 mx-1"></div>

                    <button onClick={handleFullscreen} className="p-2 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors" title="Fullscreen">
                        <Maximize size={18} />
                    </button>
                </div>
            </div>
        </div>
    )
}
