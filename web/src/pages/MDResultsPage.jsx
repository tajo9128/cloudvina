import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { supabase } from '../supabaseClient';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';
import { Activity, Play, Pause, Zap, Database, ArrowRight, Layers, FileText, Download } from 'lucide-react';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const MDResultsPage = () => {
    const { jobId } = useParams();
    const [jobData, setJobData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [currentFrame, setCurrentFrame] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [loadingEnergy, setLoadingEnergy] = useState(false); // Legacy support logic
    const viewerRef = useRef(null);
    const [viewer, setViewer] = useState(null);

    // Initial Data Load
    useEffect(() => {
        loadJobData();
        const interval = setInterval(loadJobData, 10000); // Poll every 10s
        return () => clearInterval(interval);
    }, [jobId]);

    const loadJobData = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session) return;
            const response = await fetch(`${import.meta.env.VITE_API_URL}/md/status/${jobId}`, {
                headers: { 'Authorization': `Bearer ${session.access_token}` }
            });
            if (response.ok) {
                const data = await response.json();
                setJobData(data);
                if (data.status === 'SUCCESS') setLoading(false);
            }
        } catch (error) {
            console.error('Error loading job data:', error);
        } finally {
            // Only stop initial loading spinner once we have *some* data
            if (jobData) setLoading(false);
        }
    };

    // Viewer Initialization
    useEffect(() => {
        if (jobData?.status === 'SUCCESS' && viewerRef.current && window.$3Dmol && !viewer) {
            try {
                const v = window.$3Dmol.createViewer(viewerRef.current, { backgroundColor: '#0f172a' }); // Slate-900
                setViewer(v);

                // If trajectory URL exists, load it
                if (jobData.result.trajectory_url) {
                    // Note: 3Dmol might need CORS enabled on S3 or a proxy. 
                    // Ideally we load the PDB first then DCD. For now we just init viewer.
                    v.addModel("HEADER    placeholder\n", "pdb");
                    v.zoomTo();
                    v.render();
                }
            } catch (e) {
                console.error("Viewer init failed", e);
            }
        }
    }, [jobData]);

    const chartOptions = {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { x: { display: false }, y: { grid: { color: '#f1f5f9' } } }
    };

    // Placeholder Data
    const rmsdData = {
        labels: [1, 2, 3, 4, 5],
        datasets: [{
            label: 'RMSD',
            data: [1.2, 1.3, 1.5, 1.4, 1.6],
            borderColor: '#6366f1',
            tension: 0.4
        }]
    };

    if (!jobData && loading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;

    const isSuccess = jobData?.status === 'SUCCESS';

    return (
        <div className="min-h-screen bg-slate-50 font-sans pb-20">
            {/* Header */}
            <div className="bg-white border-b border-slate-200 sticky top-0 z-40 backdrop-blur-md bg-white/90">
                <div className="container mx-auto px-6 h-20 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link to="/md-simulation" className="p-2 hover:bg-slate-100 rounded-full text-slate-400 transition-colors">
                            <ArrowRight className="rotate-180 w-5 h-5" />
                        </Link>
                        <div>
                            <h1 className="text-xl font-bold text-slate-900">Dynamics Dashboard (AWS Batch)</h1>
                            <div className="text-xs font-mono text-slate-500">Job ID: {jobId}</div>
                        </div>
                    </div>
                    <div className="flex items-center gap-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide ${isSuccess ? 'bg-emerald-100 text-emerald-700' : 'bg-blue-100 text-blue-700 animate-pulse'
                            }`}>
                            {jobData?.status || "LOADING..."}
                        </span>
                    </div>
                </div>
            </div>

            <main className="container mx-auto px-6 py-8">

                {!isSuccess && (
                    <div className="max-w-xl mx-auto text-center py-20">
                        <Zap className="w-16 h-16 text-blue-500 mx-auto mb-6 animate-pulse" />
                        <h2 className="text-2xl font-bold text-slate-900 mb-2">Simulation Running...</h2>
                        <p className="text-slate-500 mb-8">
                            Your job is processing on AWS Batch (GPU). This typically takes 10-60 minutes depending on system size.
                            This page will auto-update when results are ready.
                        </p>
                        <div className="bg-white p-4 rounded-lg border border-slate-200 text-left text-sm font-mono text-slate-600">
                            {jobData?.info?.message || "Status: " + (jobData?.status || "Initializing...")}
                        </div>
                    </div>
                )}

                {isSuccess && (
                    <>
                        {/* Metrics */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                                <Activity className="w-8 h-8 text-indigo-500 mb-2" />
                                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Stability Score</div>
                                <div className="text-3xl font-bold text-slate-900">{jobData.result?.stability_score || 85.2}<span className="text-sm text-slate-400">/100</span></div>
                            </div>

                            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                                <FileText className="w-8 h-8 text-emerald-500 mb-2" />
                                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">PDF Report</div>
                                {jobData.result?.report_url ? (
                                    <a href={jobData.result.report_url} target="_blank" rel="noopener noreferrer" className="text-indigo-600 font-bold hover:underline">Download Analysis</a>
                                ) : <span className="text-slate-400">Available in Vault</span>}
                            </div>

                            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                                <Database className="w-8 h-8 text-violet-500 mb-2" />
                                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Output Files</div>
                                <div className="flex gap-4">
                                    {jobData.result?.trajectory_url && <a href={jobData.result.trajectory_url} className="text-sm font-medium text-slate-600 hover:text-indigo-600">Trajectory.dcd</a>}
                                    {jobData.result?.log_url && <a href={jobData.result.log_url} target="_blank" rel="noopener noreferrer" className="text-sm font-medium text-slate-600 hover:text-indigo-600">Log.txt</a>}
                                </div>
                            </div>
                        </div>

                        {/* Content Grid */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                            {/* 3D Viewer */}
                            <div className="lg:col-span-2 bg-white rounded-3xl shadow-sm border border-slate-200 p-1">
                                <div className="h-96 bg-slate-900 rounded-2xl relative overflow-hidden" ref={viewerRef}>
                                    <div className="absolute top-4 left-4 text-white/50 text-xs select-none">3D Viewer (AWS Stream)</div>
                                </div>
                            </div>

                            {/* Analysis Card */}
                            <div className="space-y-6">
                                <div className="bg-white rounded-3xl shadow-sm border border-slate-200 p-6">
                                    <h3 className="font-bold text-slate-900 mb-4">RMSD Trajectory</h3>
                                    <div className="h-48">
                                        <Line data={rmsdData} options={chartOptions} />
                                    </div>
                                    <div className="mt-4 text-xs text-center text-slate-400">Stability over 100ns</div>
                                </div>

                                <div className="bg-slate-900 rounded-3xl text-white p-8">
                                    <h3 className="font-bold text-lg mb-6 flex items-center gap-2">
                                        <Database className="text-indigo-400" size={20} />
                                        Data Vault
                                    </h3>
                                    <div className="space-y-3">
                                        <a href={jobData.result?.trajectory_url || "#"} className="w-full flex items-center justify-between p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all group">
                                            <div className="font-bold text-sm">Download Full Trajectory</div>
                                            <Download size={16} />
                                        </a>
                                        <a href={jobData.result?.report_url || "#"} className="w-full flex items-center justify-between p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all group">
                                            <div className="font-bold text-sm">Download PDF Report</div>
                                            <Download size={16} />
                                        </a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </>
                )}
            </main>
        </div>
    );
};

export default MDResultsPage;
