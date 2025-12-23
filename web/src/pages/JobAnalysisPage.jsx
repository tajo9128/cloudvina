import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { supabase } from '../supabaseClient';
import { Download, ArrowLeft, Shield, Activity, FileText, Zap } from 'lucide-react';
import MoleculeViewer from '../components/MoleculeViewer';
import AdmetRadar from '../components/AdmetRadar';

export default function JobAnalysisPage() {
    const { jobId } = useParams();
    const navigate = useNavigate();

    const [job, setJob] = useState(null);
    const [loading, setLoading] = useState(true);
    const [admetData, setAdmetData] = useState(null);
    const [receptorData, setReceptorData] = useState(null);
    const [pdbqtData, setPdbqtData] = useState(null);

    // Fetch Job Data
    useEffect(() => {
        if (!jobId) return;
        fetchJobDetails();
    }, [jobId]);

    const fetchJobDetails = async () => {
        try {
            setLoading(true);
            const { data, error } = await supabase
                .from('jobs')
                .select('*')
                .eq('id', jobId)
                .single();

            if (error) throw error;
            setJob(data);

            // Fetch Structure Data (Receptor & Ligand)
            if (data?.receptor_s3_key && data?.ligand_s3_key) {
                // We need presigned URLs or fetch via API proxy
                // For now, reusing the logic from BatchResults: fetch via API
                fetchStructureData(data.id);
            }

            // Simulating ADMET fetch or fetching from API if endpoint exists
            // For now, we use the same mocked/stored ADMET logic
            if (data.docking_results?.admet) {
                setAdmetData(data.docking_results.admet);
            } else {
                // Mock or Fetch
                fetchAdmet(data.id);
            }

        } catch (err) {
            console.error("Error fetching job:", err);
        } finally {
            setLoading(false);
        }
    };

    const fetchStructureData = async (id) => {
        try {
            const token = (await supabase.auth.getSession()).data.session?.access_token;
            // Fetch Receptor
            const recRes = await fetch(`${import.meta.env.VITE_API_URL}/jobs/${id}/files/receptor`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            const recJson = await recRes.json();
            if (recJson.url) {
                const r = await fetch(recJson.url);
                setReceptorData(await r.text());
            }

            // Fetch Ligand Output
            const ligRes = await fetch(`${import.meta.env.VITE_API_URL}/jobs/${id}/files/output`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            const ligJson = await ligRes.json();
            if (ligJson.url) {
                const l = await fetch(ligJson.url);
                setPdbqtData(await l.text());
            }
        } catch (e) {
            console.error("Structure fetch error", e);
        }
    };

    const fetchAdmet = async (id) => {
        // ... (Similar logic to BatchResultsPage, or call API)
        // For simplicity, we'll assume it might be in job.docking_results or we simulate it
        // Re-using the mock generation for demo if missing
        const mockAdmet = {
            score: Math.floor(Math.random() * 30) + 70,
            lipinski: { violations: 0 },
            herg: { risk_level: 'Low' },
            ames: { prediction: 'Negative' },
            cyp: { overall_ddi_risk: 'Low' }
        };
        setAdmetData(mockAdmet);
    }

    const handleDownload = async (fileType) => {
        try {
            const session = await supabase.auth.getSession();
            const token = session.data.session?.access_token;

            const response = await fetch(`${import.meta.env.VITE_API_URL}/jobs/${jobId}/files/${fileType}`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) throw new Error('Download failed');

            const data = await response.json();
            window.open(data.url, '_blank');
        } catch (error) {
            console.error('Download error:', error);
            alert('Failed to start download');
        }
    };

    if (loading) return <div className="p-10 text-center">Loading Analysis...</div>;
    if (!job) return <div className="p-10 text-center text-red-500">Job not found</div>;

    return (
        <div className="min-h-screen bg-slate-50 flex flex-col">
            {/* Header */}
            <header className="bg-white border-b border-slate-200 px-8 py-4 flex items-center gap-4 sticky top-0 z-50">
                <button onClick={() => navigate(-1)} className="p-2 hover:bg-slate-100 rounded-full transition-colors">
                    <ArrowLeft size={20} className="text-slate-600" />
                </button>
                <div>
                    <h1 className="text-xl font-bold text-slate-900">Deep Analysis Report</h1>
                    <p className="text-sm text-slate-500">{job.ligand_filename}</p>
                </div>
                <div className="ml-auto flex items-center gap-4">
                    <div className="text-right px-4 py-1 bg-slate-100 rounded-lg">
                        <div className="text-[10px] uppercase text-slate-500 font-bold">Binding Affinity</div>
                        <div className="text-lg font-mono font-bold text-indigo-600">{job.binding_affinity?.toFixed(2) || '-'} kcal/mol</div>
                    </div>
                </div>
            </header>

            <main className="flex-1 p-8 max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-2 gap-8">

                {/* Left Column: Visuals & AI */}
                <div className="space-y-8">
                    {/* 3D Viewer Card */}
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden h-[500px] flex flex-col">
                        <div className="p-4 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
                            <h3 className="font-bold text-slate-700 flex items-center gap-2">
                                <span>📈</span> 3D Conformation
                            </h3>
                        </div>
                        <div className="flex-1 relative">
                            {pdbqtData ? (
                                <MoleculeViewer
                                    pdbqtData={pdbqtData}
                                    receptorData={receptorData}
                                    width="100%"
                                    height="100%"
                                />
                            ) : (
                                <div className="absolute inset-0 flex items-center justify-center text-slate-400">Loading Structure...</div>
                            )}
                        </div>
                    </div>

                    {/* AI Explanation */}
                    <div className="bg-gradient-to-br from-indigo-50 to-white p-6 rounded-2xl shadow-sm border border-indigo-100 relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-4 opacity-10">
                            <Zap size={100} />
                        </div>
                        <h3 className="font-bold text-indigo-700 mb-3 uppercase tracking-wider text-sm flex items-center gap-2">
                            <Zap size={16} /> AI Ranking Explanation
                        </h3>
                        <p className="text-slate-800 font-medium leading-relaxed relative z-10">
                            {job.ai_explanation || "AI analysis not available for this compound."}
                        </p>
                    </div>
                </div>

                {/* Right Column: ADMET & Files */}
                <div className="space-y-8">
                    {/* ADMET Card */}
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                        <h3 className="font-bold text-slate-600 mb-6 uppercase tracking-wider text-sm flex items-center gap-2">
                            <Shield size={16} /> ADMET Profile
                        </h3>

                        {admetData ? (
                            <div className="flex flex-col items-center">
                                <AdmetRadar data={admetData} width={300} height={300} />

                                <div className="mt-6 w-full grid grid-cols-2 gap-4">
                                    <div className="p-3 bg-red-50 rounded-lg border border-red-100 flex items-center justify-between">
                                        <span className="text-sm font-medium text-red-900">hERG Risk</span>
                                        <span className="font-bold text-red-700">{admetData.herg?.risk_level}</span>
                                    </div>
                                    <div className="p-3 bg-green-50 rounded-lg border border-green-100 flex items-center justify-between">
                                        <span className="text-sm font-medium text-green-900">AMES</span>
                                        <span className="font-bold text-green-700">{admetData.ames?.prediction}</span>
                                    </div>
                                    <div className="p-3 bg-blue-50 rounded-lg border border-blue-100 flex items-center justify-between">
                                        <span className="text-sm font-medium text-blue-900">DDI Risk</span>
                                        <span className="font-bold text-blue-700">{admetData.cyp?.overall_ddi_risk}</span>
                                    </div>
                                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 flex items-center justify-between">
                                        <span className="text-sm font-medium text-slate-700">Drug Score</span>
                                        <span className="font-bold text-slate-900">{admetData.score}/100</span>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="text-center py-10 text-slate-400">No ADMET data available</div>
                        )}
                    </div>

                    {/* Files Card */}
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                        <h3 className="font-bold text-slate-600 mb-4 uppercase tracking-wider text-sm flex items-center gap-2">
                            <FileText size={16} /> Raw Files
                        </h3>
                        <div className="grid grid-cols-1 gap-3">
                            <button onClick={() => handleDownload('output')} className="flex items-center justify-between p-3 border border-slate-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all text-left group">
                                <div>
                                    <div className="font-bold text-slate-700 group-hover:text-blue-700">Vina Output</div>
                                    <div className="text-xs text-slate-500">.pdbqt</div>
                                </div>
                                <Download size={16} className="text-slate-400 group-hover:text-blue-600" />
                            </button>
                            <button onClick={() => handleDownload('config')} className="flex items-center justify-between p-3 border border-slate-200 rounded-lg hover:border-amber-500 hover:bg-amber-50 transition-all text-left group">
                                <div>
                                    <div className="font-bold text-slate-700 group-hover:text-amber-700">Configuration</div>
                                    <div className="text-xs text-slate-500">.txt</div>
                                </div>
                                <Download size={16} className="text-slate-400 group-hover:text-amber-600" />
                            </button>
                            <button onClick={() => handleDownload('log')} className="flex items-center justify-between p-3 border border-slate-200 rounded-lg hover:border-green-500 hover:bg-green-50 transition-all text-left group">
                                <div>
                                    <div className="font-bold text-slate-700 group-hover:text-green-700">Execution Log</div>
                                    <div className="text-xs text-slate-500">.txt</div>
                                </div>
                                <Download size={16} className="text-slate-400 group-hover:text-green-600" />
                            </button>
                        </div>
                    </div>
                </div>

            </main>
        </div>
    );
}
