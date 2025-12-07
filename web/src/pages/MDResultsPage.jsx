import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { supabase } from '../supabaseClient';

const MDResultsPage = () => {
    const { jobId } = useParams();
    const [jobData, setJobData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [currentFrame, setCurrentFrame] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const viewerRef = useRef(null);
    const [viewer, setViewer] = useState(null);
    const [loadingEnergy, setLoadingEnergy] = useState(false);

    // Initial check for existing energy result
    useEffect(() => {
        if (jobData?.result?.analysis?.binding_energy) {
            setLoadingEnergy(false);
        }
    }, [jobData]);

    const calculateBindingEnergy = async () => {
        try {
            setLoadingEnergy(true);
            const { data: { session } } = await supabase.auth.getSession();
            if (!session) return;

            // Trigger calculation
            const response = await fetch(
                `${import.meta.env.VITE_API_URL}/md/analyze/binding-energy/${jobId}`,
                {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${session.access_token}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ ligand_resname: 'LIG' })
                }
            );

            if (response.ok) {
                // Poll for status
                pollEnergyStatus(session.access_token);
            } else {
                setLoadingEnergy(false);
                alert("Failed to start analysis.");
            }
        } catch (error) {
            console.error(error);
            setLoadingEnergy(false);
        }
    };

    const pollEnergyStatus = async (token) => {
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`${import.meta.env.VITE_API_URL}/md/status/${jobId}_binding_energy`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                });

                if (res.ok) {
                    const data = await res.json();
                    if (data.status === 'SUCCESS' && data.result) {
                        clearInterval(interval);
                        setLoadingEnergy(false);
                        // Update local jobData with new analysis
                        setJobData(prev => ({
                            ...prev,
                            result: {
                                ...prev.result,
                                analysis: {
                                    ...prev.result.analysis,
                                    binding_energy: data.result.binding_energy,
                                    std_dev: data.result.std_dev
                                }
                            }
                        }));
                    } else if (data.status === 'FAILURE') {
                        clearInterval(interval);
                        setLoadingEnergy(false);
                        alert("Analysis failed: " + data.error);
                    }
                }
            } catch (err) {
                console.error(err);
            }
        }, 5000); // Poll every 5s
    };

    useEffect(() => {
        loadJobData();
        initializeViewer();
    }, [jobId]);

    const loadJobData = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session) return;

            const response = await fetch(
                `${import.meta.env.VITE_API_URL}/md/status/${jobId}`,
                {
                    headers: {
                        'Authorization': `Bearer ${session.access_token}`
                    }
                }
            );

            if (response.ok) {
                const data = await response.json();
                setJobData(data);
            }
        } catch (error) {
            console.error('Error loading job data:', error);
        } finally {
            setLoading(false);
        }
    };

    const initializeViewer = () => {
        if (viewerRef.current && window.$3Dmol) {
            const v = window.$3Dmol.createViewer(viewerRef.current, {
                backgroundColor: 'white'
            });
            setViewer(v);
        }
    };

    const togglePlayback = () => {
        setIsPlaying(!isPlaying);
    };

    useEffect(() => {
        if (isPlaying && jobData?.result?.total_frames) {
            const interval = setInterval(() => {
                setCurrentFrame(prev =>
                    prev >= jobData.result.total_frames - 1 ? 0 : prev + 1
                );
            }, 100);
            return () => clearInterval(interval);
        }
    }, [isPlaying, jobData]);

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-xl">Loading results...</div>
            </div>
        );
    }

    if (!jobData || jobData.status !== 'SUCCESS') {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-center">
                    <h2 className="text-2xl font-bold text-gray-900 mb-2">
                        Results Not Available
                    </h2>
                    <p className="text-gray-600">
                        {jobData?.status === 'FAILURE'
                            ? 'Simulation failed. Please try again.'
                            : 'Simulation in progress or not found.'}
                    </p>
                </div>
            </div>
        );
    }

    const analysis = jobData.result?.analysis || {};

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 py-12 px-4">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-4xl font-bold text-gray-900 mb-2">
                        MD Simulation Results
                    </h1>
                    <p className="text-gray-600">Job ID: {jobId}</p>
                    <p className="text-sm text-gray-500">
                        Duration: {jobData.result?.duration_ns?.toFixed(2)} ns |
                        Frames: {jobData.result?.total_frames}
                    </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Main Viewer */}
                    <div className="lg:col-span-2">
                        <div className="bg-white rounded-2xl shadow-xl p-6">
                            <h2 className="text-2xl font-bold text-gray-900 mb-4">
                                Trajectory Viewer
                            </h2>

                            {/* 3Dmol Viewer */}
                            <div
                                ref={viewerRef}
                                className="w-full h-96 border-2 border-gray-200 rounded-lg mb-4"
                                style={{ position: 'relative' }}
                            />

                            {/* Playback Controls */}
                            <div className="space-y-4">
                                <div className="flex items-center gap-4">
                                    <button
                                        onClick={togglePlayback}
                                        className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold"
                                    >
                                        {isPlaying ? '⏸ Pause' : '▶ Play'}
                                    </button>
                                    <span className="text-sm text-gray-600">
                                        Frame: {currentFrame} / {jobData.result?.total_frames || 0}
                                    </span>
                                </div>

                                {/* Frame Slider */}
                                <input
                                    type="range"
                                    min="0"
                                    max={(jobData.result?.total_frames || 1) - 1}
                                    value={currentFrame}
                                    onChange={(e) => setCurrentFrame(parseInt(e.target.value))}
                                    className="w-full"
                                />

                                {/* Style Controls */}
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => viewer?.setStyle({}, { cartoon: { color: 'spectrum' } })}
                                        className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 text-sm"
                                    >
                                        Cartoon
                                    </button>
                                    <button
                                        onClick={() => viewer?.setStyle({}, { stick: {} })}
                                        className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 text-sm"
                                    >
                                        Stick
                                    </button>
                                    <button
                                        onClick={() => viewer?.setStyle({}, { sphere: {} })}
                                        className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 text-sm"
                                    >
                                        Sphere
                                    </button>
                                </div>
                            </div>

                            {/* Note about trajectory access */}
                            <div className="mt-4 bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                                <p className="text-sm text-blue-800">
                                    <strong>📌 Note:</strong> Full trajectory playback requires the DCD file.
                                    Currently showing structure visualization. Download the trajectory from your Colab session for complete analysis.
                                </p>
                            </div>
                        </div>

                        {/* Analysis Plots */}
                        {analysis.plots && (
                            <div className="mt-6 bg-white rounded-2xl shadow-xl p-6">
                                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                                    Analysis Plots
                                </h2>
                                <div className="grid grid-cols-2 gap-4">
                                    {analysis.plots.rmsd && (
                                        <div>
                                            <h3 className="font-semibold text-gray-700 mb-2">RMSD</h3>
                                            <p className="text-sm text-gray-500">Available in Colab output</p>
                                        </div>
                                    )}
                                    {analysis.plots.rmsf && (
                                        <div>
                                            <h3 className="font-semibold text-gray-700 mb-2">RMSF</h3>
                                            <p className="text-sm text-gray-500">Available in Colab output</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Analysis Sidebar */}
                    <div className="space-y-6">
                        {/* Trajectory Metrics */}
                        <div className="bg-white rounded-2xl shadow-xl p-6">
                            <h3 className="text-xl font-bold text-gray-900 mb-4">
                                Trajectory Metrics
                            </h3>
                            <div className="space-y-3">
                                {analysis.rmsd_mean && (
                                    <div className="flex justify-between border-b pb-2">
                                        <span className="text-gray-600">RMSD (mean)</span>
                                        <span className="font-semibold">{analysis.rmsd_mean.toFixed(2)} Å</span>
                                    </div>
                                )}
                                {analysis.rmsd_max && (
                                    <div className="flex justify-between border-b pb-2">
                                        <span className="text-gray-600">RMSD (max)</span>
                                        <span className="font-semibold">{analysis.rmsd_max.toFixed(2)} Å</span>
                                    </div>
                                )}
                                {analysis.mean_flexibility && (
                                    <div className="flex justify-between border-b pb-2">
                                        <span className="text-gray-600">Avg Flexibility</span>
                                        <span className="font-semibold">{analysis.mean_flexibility.toFixed(2)} Å</span>
                                    </div>
                                )}
                                {analysis.flexible_residues !== undefined && (
                                    <div className="flex justify-between border-b pb-2">
                                        <span className="text-gray-600">Flexible Residues</span>
                                        <span className="font-semibold">{analysis.flexible_residues}</span>
                                    </div>
                                )}
                                {analysis.rg_mean && (
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Radius of Gyration</span>
                                        <span className="font-semibold">{analysis.rg_mean.toFixed(2)} Å</span>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Binding Energy Analysis (Phase 4) */}
                        <div className="bg-white rounded-2xl shadow-xl p-6">
                            <h3 className="text-xl font-bold text-gray-900 mb-4">
                                Binding Free Energy
                            </h3>

                            {!jobData.result?.analysis?.binding_energy ? (
                                <div className="text-center">
                                    <p className="text-sm text-gray-600 mb-4">
                                        Estimate binding affinity using MM-GBSA (Implicit Solvent).
                                    </p>

                                    {loadingEnergy ? (
                                        <div className="flex flex-col items-center py-2">
                                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mb-2"></div>
                                            <span className="text-sm text-gray-500">Calculating... this may take 1-2 mins</span>
                                        </div>
                                    ) : (
                                        <button
                                            onClick={calculateBindingEnergy}
                                            disabled={loadingEnergy}
                                            className="w-full px-4 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-xl font-bold hover:shadow-lg transition-all transform hover:-translate-y-0.5"
                                        >
                                            ⚡ Calculate ΔG (MM-GBSA)
                                        </button>
                                    )}
                                </div>
                            ) : (
                                <div className="space-y-3">
                                    <div className="flex justify-between items-end border-b pb-2">
                                        <span className="text-gray-600">ΔG (Binding)</span>
                                        <div className="text-right">
                                            <span className="text-2xl font-bold text-green-600">
                                                {jobData.result.analysis.binding_energy.toFixed(2)}
                                            </span>
                                            <span className="text-xs text-gray-500 ml-1">kcal/mol</span>
                                        </div>
                                    </div>
                                    <div className="flex justify-between items-center text-sm">
                                        <span className="text-gray-500">Std Dev</span>
                                        <span className="font-mono text-gray-700">± {jobData.result.analysis.std_dev?.toFixed(2)}</span>
                                    </div>
                                    <div className="mt-2 bg-green-50 text-green-800 text-xs p-2 rounded border border-green-100">
                                        <strong>Interpretation:</strong> Lower (more negative) values indicate stronger binding.
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Interaction Analysis */}
                        <div className="bg-white rounded-2xl shadow-xl p-6">
                            <h3 className="text-xl font-bold text-gray-900 mb-4">
                                Protein-Ligand Interactions
                            </h3>
                            <div className="space-y-3">
                                {analysis.hydrogen_bonds !== undefined && (
                                    <div className="flex items-center justify-between">
                                        <span className="text-gray-600">🔗 H-Bonds</span>
                                        <span className="font-semibold text-blue-600 text-lg">
                                            {analysis.hydrogen_bonds}
                                        </span>
                                    </div>
                                )}
                                {analysis.hydrophobic_contacts !== undefined && (
                                    <div className="flex items-center justify-between">
                                        <span className="text-gray-600">💧 Hydrophobic</span>
                                        <span className="font-semibold text-green-600 text-lg">
                                            {analysis.hydrophobic_contacts}
                                        </span>
                                    </div>
                                )}
                                {analysis.pi_stacking !== undefined && (
                                    <div className="flex items-center justify-between">
                                        <span className="text-gray-600">🔺 Pi-Stacking</span>
                                        <span className="font-semibold text-purple-600 text-lg">
                                            {analysis.pi_stacking}
                                        </span>
                                    </div>
                                )}
                                {analysis.salt_bridges !== undefined && (
                                    <div className="flex items-center justify-between">
                                        <span className="text-gray-600">⚡ Salt Bridges</span>
                                        <span className="font-semibold text-orange-600 text-lg">
                                            {analysis.salt_bridges}
                                        </span>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Download Options */}
                        <div className="bg-white rounded-2xl shadow-xl p-6">
                            <h3 className="text-xl font-bold text-gray-900 mb-4">
                                Download Results
                            </h3>
                            <div className="space-y-2">
                                <button className="w-full px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 text-sm">
                                    📊 Analysis Summary (CSV)
                                </button>
                                <button className="w-full px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 text-sm">
                                    📈 RMSD Plot (PNG)
                                </button>
                                <button className="w-full px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 text-sm">
                                    📉 RMSF Plot (PNG)
                                </button>
                                <button className="w-full px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 text-sm">
                                    🎬 Trajectory (DCD)
                                </button>
                            </div>
                            <p className="text-xs text-gray-500 mt-3">
                                Files are available in your Colab session output
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MDResultsPage;
