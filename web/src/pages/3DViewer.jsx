import React, { useEffect, useRef, useState } from 'react';
import * as $3Dmol from '3dmol/build/3Dmol.js';
import { useParams, Link } from 'react-router-dom';
import { API_URL } from '../config';
import { supabase } from '../supabaseClient';

const ThreeDViewer = () => {
    const { jobId } = useParams();
    const viewerRef = useRef(null);
    const [viewer, setViewer] = useState(null);
    const [status, setStatus] = useState('Ready');
    const [bestScore, setBestScore] = useState(null);
    const [generation, setGeneration] = useState(0);
    const [mode, setMode] = useState(jobId ? 'job' : 'manual');

    // Interaction Visualization State (NEW)
    const [interactions, setInteractions] = useState(null);
    const [showHBonds, setShowHBonds] = useState(true);
    const [showHydrophobic, setShowHydrophobic] = useState(false);

    // Cavity Visualization State (NEW)
    const [cavities, setCavities] = useState(null);
    const [showCavities, setShowCavities] = useState(true);

    // Initialize 3Dmol Viewer
    useEffect(() => {
        if (!viewerRef.current) return;

        const v = new $3Dmol.GLViewer(viewerRef.current, {
            backgroundColor: 'white',
            id: 'gldiv'
        });
        setViewer(v);
        v.zoomTo();
        v.render();

        return () => {
            // Cleanup
        };
    }, []);

    // Load Job Data
    useEffect(() => {
        if (!jobId || !viewer) return;

        const loadJobData = async () => {
            setStatus('Loading Job Data...');
            try {
                const { data: { session } } = await supabase.auth.getSession();
                if (!session) throw new Error('Not authenticated');

                const res = await fetch(`${API_URL}/jobs/${jobId}`, {
                    headers: { 'Authorization': `Bearer ${session.access_token}` }
                });
                if (!res.ok) throw new Error('Failed to fetch job');
                const job = await res.json();

                if (job.status === 'SUCCEEDED' && job.download_urls?.output) {
                    const pdbqtRes = await fetch(job.download_urls.output);
                    const pdbqtText = await pdbqtRes.text();

                    viewer.removeAllModels();
                    viewer.addModel(pdbqtText, "pdbqt");
                    viewer.setStyle({}, { stick: { colorscheme: "greenCarbon" } });
                    viewer.zoomTo();
                    viewer.render();
                    setStatus(`Loaded Job: ${job.job_id.slice(0, 8)}`);

                    // Fetch interactions for H-bond visualization
                    try {
                        const intRes = await fetch(`${API_URL}/jobs/${jobId}/interactions`, {
                            headers: { 'Authorization': `Bearer ${session.access_token}` }
                        });
                        if (intRes.ok) {
                            const intData = await intRes.json();
                            if (intData.interactions) {
                                setInteractions(intData.interactions);
                                console.log('Loaded interactions:', intData.interactions);
                            }
                        }
                    } catch (intErr) {
                        console.warn('Failed to load interactions:', intErr);
                    }

                    // Fetch cavities for pocket visualization
                    try {
                        const cavRes = await fetch(`${API_URL}/jobs/${jobId}/detect-cavities`, {
                            headers: { 'Authorization': `Bearer ${session.access_token}` }
                        });
                        if (cavRes.ok) {
                            const cavData = await cavRes.json();
                            if (cavData.cavities) {
                                setCavities(cavData.cavities);
                                console.log('Loaded cavities:', cavData.cavities);
                            }
                        }
                    } catch (cavErr) {
                        console.warn('Failed to load cavities:', cavErr);
                    }
                } else {
                    setStatus(`Job Status: ${job.status}`);
                }
            } catch (err) {
                console.error(err);
                setStatus(`Error: ${err.message}`);
            }
        };

        loadJobData();
    }, [jobId, viewer]);

    // Draw interactions and cavities when toggle changes
    useEffect(() => {
        if (!viewer) return;

        // Clear existing shapes
        viewer.removeAllShapes();

        // Draw H-bonds as cyan dashed lines
        if (showHBonds && interactions?.hydrogen_bonds) {
            interactions.hydrogen_bonds.forEach(bond => {
                if (bond.ligand_coords && bond.protein_coords) {
                    viewer.addCylinder({
                        start: { x: bond.ligand_coords[0], y: bond.ligand_coords[1], z: bond.ligand_coords[2] },
                        end: { x: bond.protein_coords[0], y: bond.protein_coords[1], z: bond.protein_coords[2] },
                        radius: 0.08,
                        color: '#2563eb',
                        dashed: true
                    });
                }
            });
        }

        // Draw hydrophobic contacts as yellow dotted lines
        if (showHydrophobic && interactions?.hydrophobic_contacts) {
            interactions.hydrophobic_contacts.forEach(contact => {
                if (contact.ligand_coords && contact.protein_coords) {
                    viewer.addCylinder({
                        start: { x: contact.ligand_coords[0], y: contact.ligand_coords[1], z: contact.ligand_coords[2] },
                        end: { x: contact.protein_coords[0], y: contact.protein_coords[1], z: contact.protein_coords[2] },
                        radius: 0.05,
                        color: '#eab308',
                        dashed: true
                    });
                }
            });
        }

        // Draw cavities as colored spheres
        if (showCavities && cavities) {
            const colors = ['#22c55e', '#f97316', '#a855f7', '#ec4899', '#14b8a6'];
            cavities.forEach((cavity, i) => {
                viewer.addSphere({
                    center: { x: cavity.center_x, y: cavity.center_y, z: cavity.center_z },
                    radius: Math.min(cavity.size_x, cavity.size_y, cavity.size_z) / 4,
                    color: colors[i % colors.length],
                    opacity: 0.3
                });
                viewer.addLabel(`Pocket ${cavity.pocket_id}`, {
                    position: { x: cavity.center_x, y: cavity.center_y + 3, z: cavity.center_z },
                    backgroundColor: colors[i % colors.length],
                    fontColor: 'white',
                    fontSize: 12
                });
            });
        }

        viewer.render();
    }, [viewer, interactions, cavities, showHBonds, showHydrophobic, showCavities]);

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (!file || !viewer) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target.result;
            const ext = file.name.split('.').pop().toLowerCase();

            viewer.removeAllModels();
            viewer.addModel(content, ext);
            viewer.setStyle({}, { stick: { colorscheme: "greenCarbon" } });
            viewer.zoomTo();
            viewer.render();
            setStatus(`Loaded File: ${file.name}`);
        };
        reader.readAsText(file);
    };

    return (
        <div className="min-h-screen bg-slate-50 pb-20">
            {/* Header Section */}
            <div className="bg-white border-b border-slate-200 pt-4 pb-6">
                <div className="container mx-auto px-4">
                    <div className="flex justify-between items-center">
                        <div>
                            <h1 className="text-3xl font-bold text-slate-900 mb-2">3D Viewer</h1>
                            <p className="text-slate-500">Interactive molecular visualization and analysis</p>
                        </div>
                        <div className="flex gap-3">
                            {jobId && (
                                <Link to={`/dock/${jobId}`} className="btn-secondary">
                                    &larr; Back to Job
                                </Link>
                            )}
                            <Link to="/dashboard" className="btn-secondary">
                                Dashboard
                            </Link>
                        </div>
                    </div>
                </div>
            </div>

            <div className="container mx-auto px-4 py-8">
                <div className="flex flex-col lg:flex-row gap-6 h-[800px]">

                    {/* Sidebar Controls */}
                    <div className="w-full lg:w-1/4 flex flex-col gap-6">
                        {/* Status Card */}
                        <div className="card p-6">
                            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">System Status</h2>
                            <div className="flex items-center gap-3">
                                <span className="relative flex h-3 w-3">
                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                    <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                                </span>
                                <span className="font-medium text-slate-700 truncate" title={status}>{status}</span>
                            </div>
                        </div>

                        {/* Upload Card */}
                        <div className="card p-6">
                            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">Manual Upload</h2>
                            <label className="block">
                                <span className="sr-only">Choose file</span>
                                <input
                                    type="file"
                                    accept=".pdb,.sdf,.mol2,.pdbqt"
                                    onChange={handleFileUpload}
                                    className="block w-full text-sm text-slate-500
                                    file:mr-4 file:py-2.5 file:px-4
                                    file:rounded-lg file:border-0
                                    file:text-sm file:font-semibold
                                    file:bg-primary-50 file:text-primary-700
                                    hover:file:bg-primary-100
                                    cursor-pointer transition-colors"
                                />
                            </label>
                            <p className="text-xs text-slate-400 mt-3">
                                Supports .pdb, .sdf, .mol2, .pdbqt
                            </p>
                        </div>

                        {/* Interaction Display Controls (NEW) */}
                        {interactions && (
                            <div className="card p-6">
                                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">
                                    Interactions
                                </h2>
                                <div className="space-y-3">
                                    <label className="flex items-center justify-between cursor-pointer">
                                        <span className="flex items-center gap-2">
                                            <span className="w-3 h-3 rounded-full bg-cyan-400"></span>
                                            <span className="text-sm text-slate-700">H-Bonds</span>
                                            <span className="text-xs bg-cyan-100 text-cyan-700 px-1.5 py-0.5 rounded">
                                                {interactions.hydrogen_bonds?.length || 0}
                                            </span>
                                        </span>
                                        <input
                                            type="checkbox"
                                            checked={showHBonds}
                                            onChange={(e) => setShowHBonds(e.target.checked)}
                                            className="w-4 h-4 text-cyan-600 rounded border-slate-300 focus:ring-cyan-500"
                                        />
                                    </label>
                                    <label className="flex items-center justify-between cursor-pointer">
                                        <span className="flex items-center gap-2">
                                            <span className="w-3 h-3 rounded-full bg-amber-400"></span>
                                            <span className="text-sm text-slate-700">Hydrophobic</span>
                                            <span className="text-xs bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded">
                                                {interactions.hydrophobic_contacts?.length || 0}
                                            </span>
                                        </span>
                                        <input
                                            type="checkbox"
                                            checked={showHydrophobic}
                                            onChange={(e) => setShowHydrophobic(e.target.checked)}
                                            className="w-4 h-4 text-amber-600 rounded border-slate-300 focus:ring-amber-500"
                                        />
                                    </label>
                                </div>
                                <p className="text-xs text-slate-400 mt-3">
                                    Key residues: {interactions.residues_involved?.slice(0, 5).join(', ')}
                                </p>
                            </div>
                        )}

                        {/* Cavity Display Controls (NEW) */}
                        {cavities && (
                            <div className="card p-6">
                                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">
                                    Binding Pockets
                                </h2>
                                <div className="space-y-3">
                                    <label className="flex items-center justify-between cursor-pointer">
                                        <span className="flex items-center gap-2">
                                            <span className="w-3 h-3 rounded-full bg-green-400"></span>
                                            <span className="text-sm text-slate-700">Show Cavities</span>
                                            <span className="text-xs bg-green-100 text-green-700 px-1.5 py-0.5 rounded">
                                                {cavities.length}
                                            </span>
                                        </span>
                                        <input
                                            type="checkbox"
                                            checked={showCavities}
                                            onChange={(e) => setShowCavities(e.target.checked)}
                                            className="w-4 h-4 text-green-600 rounded border-slate-300 focus:ring-green-500"
                                        />
                                    </label>
                                </div>
                                <p className="text-xs text-slate-400 mt-3">
                                    Detected {cavities.length} potential binding {cavities.length === 1 ? 'site' : 'sites'}
                                </p>
                            </div>
                        )}
                        {generation > 0 && (
                            <div className="card p-6">
                                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">Evolution Stats</h2>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 text-center">
                                        <div className="text-xs text-slate-500 uppercase mb-1">Gen</div>
                                        <div className="text-2xl font-bold text-slate-900">{generation}</div>
                                    </div>
                                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 text-center">
                                        <div className="text-xs text-slate-500 uppercase mb-1">Score</div>
                                        <div className="text-2xl font-bold text-primary-600">
                                            {bestScore ? bestScore.toFixed(2) : '-'}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Main Viewer Area */}
                    <div className="w-full lg:w-3/4 card relative overflow-hidden bg-white">
                        <div
                            ref={viewerRef}
                            className="w-full h-full"
                            style={{ outline: 'none' }}
                        />

                        {/* Overlay Branding */}
                        <div className="absolute bottom-4 right-4 bg-white/90 backdrop-blur px-3 py-1.5 rounded-lg shadow-sm border border-slate-200">
                            <p className="text-xs font-medium text-slate-500">
                                Powered by <span className="text-primary-600">3Dmol.js</span>
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ThreeDViewer;
