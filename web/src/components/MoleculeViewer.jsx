import { useEffect, useRef, useState } from 'react'
import * as $3Dmol from '3dmol/build/3Dmol.js'
import { Eye, RotateCw, Layers, Type, Maximize, Focus, Palette, BookOpen, Camera, RefreshCw } from 'lucide-react'

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
                receptor: { style: 'cartoon', color: 'ss', opacity: 1.0 },  // Secondary structure coloring (rainbow)
                ligand: { style: 'stick', color: 'greenCarbon', radius: 0.25 },
                surface: false,
                hbonds: true
            },
            analysis: {
                receptor: { style: 'cartoon', color: 'ss', opacity: 0.8 },  // Secondary structure coloring
                ligand: { style: 'stick', color: 'cyanCarbon', radius: 0.25 },
                surface: false,
                hbonds: true
            },
            surface: {
                receptor: { style: 'cartoon', color: 'ss', opacity: 0.7 },  // Colored receptor for surface mode
                ligand: { style: 'stick', color: 'greenCarbon', radius: 0.25 },
                surface: true,
                hbonds: false
            }
        }[currentPreset]

        // --- ADD RECEPTOR ---
        if (receptorData && isValidPDB(receptorData)) {
            const m = viewer.addModel(receptorData, receptorType)

            // 🌟 CRITICAL ENHANCEMENT: Auto-calculate Secondary Structure
            // This allows PDBQT (which lacks SS info) to be rendered as beautiful Caroons
            // instead of falling back to messy Sticks or Lines.
            try {
                m.calculateSecondaryStructure({
                    hbondCutoff: 3.5,
                    anchorLength: 2,
                    alphaHelix: { minLength: 4 },
                    betaSheet: { minLength: 2 }
                })
            } catch (e) {
                console.warn("MoleculeViewer: SS Calculation failed", e)
            }

            // Refined Styling Logic
            let style = {}
            if (config.receptor.style === 'cartoon') {
                // Now we can use Cartoon for everything!
                style.cartoon = {
                    colorscheme: 'spectrum', // Rainbow (Blue->Red N->C)
                    opacity: config.receptor.opacity,
                    thickness: 0.8, // Slightly thicker for "Premium" look
                    ribbon: false
                }
            } else {
                style[config.receptor.style] = {
                    colorscheme: config.receptor.color,
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

    const handleSnapshot = () => {
        if (!viewerRef.current) return
        try {
            const dataURL = viewerRef.current.pngURI()
            const link = document.createElement('a')
            link.href = dataURL
            link.download = `biodockify-structure-${Date.now()}.png`
            link.click()
        } catch (e) {
            console.error("Snapshot failed", e)
        }
    }

    const handleResetView = () => {
        if (!viewerRef.current) return
        viewerRef.current.zoomTo({ hetflag: true }, 1000)
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

            {/* Bottom Controls Bar (Always Visible) */}
            <div className="absolute bottom-0 left-0 right-0 bg-white/95 backdrop-blur border-t border-slate-200 p-3 flex flex-col gap-3 z-20">

                {/* Row 1: Main Controls */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <h4 className="text-xs font-bold text-slate-400 uppercase mr-1">View</h4>
                        <button onClick={() => setCurrentPreset('publication')} className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all border ${currentPreset === 'publication' ? 'bg-indigo-50 border-indigo-200 text-indigo-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}>
                            Default
                        </button>
                        <button onClick={() => setCurrentPreset('surface')} className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all border ${currentPreset === 'surface' ? 'bg-emerald-50 border-emerald-200 text-emerald-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}>
                            Surface
                        </button>
                        <button onClick={() => setCurrentPreset('analysis')} className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all border ${currentPreset === 'analysis' ? 'bg-blue-50 border-blue-200 text-blue-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}>
                            Analysis
                        </button>
                    </div>

                    <div className="flex items-center gap-2">
                        <button onClick={() => isSpinning ? setIsSpinning(false) : setIsSpinning(true)} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all border ${isSpinning ? 'bg-indigo-600 border-indigo-600 text-white' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}>
                            <RotateCw size={14} className={isSpinning ? 'animate-spin' : ''} />
                            {isSpinning ? 'Spinning' : 'Spin'}
                        </button>
                        <button onClick={handleResetView} className="p-2 rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-50 hover:text-indigo-600 transition-colors" title="Reset Camera">
                            <Focus size={16} />
                        </button>
                        <button onClick={handleFullscreen} className="p-2 rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-50 hover:text-indigo-600 transition-colors" title="Fullscreen">
                            <Maximize size={16} />
                        </button>
                    </div>
                </div>

                {/* Row 2: Toggles */}
                <div className="flex items-center gap-4 border-t border-slate-100 pt-2">
                    <label className="flex items-center gap-2 cursor-pointer group">
                        <div className={`w-9 h-5 rounded-full p-0.5 transition-colors ${showLabels ? 'bg-indigo-500' : 'bg-slate-200'}`} onClick={() => setShowLabels(!showLabels)}>
                            <div className={`w-4 h-4 bg-white rounded-full shadow-sm transition-transform ${showLabels ? 'translate-x-full' : ''}`} />
                        </div>
                        <span className="text-xs font-medium text-slate-600 group-hover:text-indigo-600 transition-colors">Labels</span>
                    </label>

                    <label className="flex items-center gap-2 cursor-pointer group">
                        <div className={`w-9 h-5 rounded-full p-0.5 transition-colors ${showHBonds ? 'bg-indigo-500' : 'bg-slate-200'}`} onClick={() => setShowHBonds(!showHBonds)}>
                            <div className={`w-4 h-4 bg-white rounded-full shadow-sm transition-transform ${showHBonds ? 'translate-x-full' : ''}`} />
                        </div>
                        <span className="text-xs font-medium text-slate-600 group-hover:text-indigo-600 transition-colors">H-Bonds</span>
                    </label>

                    <div className="flex-1 text-right">
                        <div className="flex-1 text-right">
                            <button onClick={handleSnapshot} className="px-3 py-1.5 bg-slate-900 text-white rounded-lg text-xs font-bold hover:bg-slate-800 transition-all flex items-center gap-2 ml-auto shadow-sm">
                                <Camera size={14} /> Take Snapshot
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
