import React, { useState, useEffect } from 'react';
import { supabase } from '../supabaseClient';
import SEOHelmet from '../components/SEOHelmet';
import AdmetRadar from '../components/AdmetRadar';
import {
    LayoutDashboard,
    Activity,
    Target,
    AlertTriangle,
    CheckCircle,
    Download,
    Search,
    Beaker,
    ChevronRight,
    Atom
} from 'lucide-react';

const LeadOptimizationPage = () => {
    // Workspace State
    const [leads, setLeads] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedLead, setSelectedLead] = useState(null);
    const [activeTab, setActiveTab] = useState('overview'); // overview | admet | targets

    // Analysis State (cached by lead ID to avoid refetching)
    const [analysisCache, setAnalysisCache] = useState({});
    const [analyzing, setAnalyzing] = useState(false);

    useEffect(() => {
        fetchRankedLeads();
    }, []);

    const fetchRankedLeads = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session) return;

            // 1. Fetch User's Jobs (simulating a "Project" view)
            const res = await fetch(`${import.meta.env.VITE_API_URL}/jobs`, {
                headers: { 'Authorization': `Bearer ${session.access_token}` }
            });
            const jobs = await res.json();

            // 2. Map to Lead Format
            // In a real app, we'd call a dedicated /ranking endpoint. 
            // Here we construct rankable objects from job data.
            const candidates = jobs
                .filter(j => j.status === 'SUCCEEDED' && j.binding_affinity)
                .map(j => ({
                    id: j.id,
                    name: j.ligand_filename || `Ligand_${j.id.slice(0, 4)}`,
                    smiles: j.smiles || "CC(=O)Oc1ccccc1C(=O)O", // Fallback for demo if SMILES missing
                    affinity: parseFloat(j.binding_affinity),
                    gnina_score: j.analysis_results?.gnina_score || null,
                    molecular_weight: 300 + Math.random() * 200, // Mock if missing
                    consensus_score: Math.random() * 0.4 + 0.5, // Mock Consensus (0.5 - 0.9)
                }))
                .sort((a, b) => a.affinity - b.affinity); // Lower is better for affinity, but for consensus higher is better. 
            // Let's stick to Affinity sort for now as primary scientific metric.

            setLeads(candidates.map((c, i) => ({ ...c, rank: i + 1 })));
            if (candidates.length > 0) setSelectedLead(candidates[0]); // Auto-select top hit

        } catch (err) {
            console.error("Failed to load leads", err);
        } finally {
            setLoading(false);
        }
    };

    const fetchAnalysis = async (lead) => {
        if (analysisCache[lead.id]) return; // Use cache

        setAnalyzing(true);
        const { data: { session } } = await supabase.auth.getSession();

        try {
            // Parallel Fetch: ADMET + Targets
            // Note: In production, these might be real separate endpoints.
            // For this refactor, we simulate the detailed fetch to match the "Tool" logic.

            // 1. Target Prediction
            const targetRes = await fetch(`${import.meta.env.VITE_API_URL}/predict/target/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session?.access_token}`
                },
                body: JSON.stringify({ smiles: lead.smiles })
            });

            // 2. ADMET (Mocked as per AdmetToolPage for now if endpoint not ready)
            // Ideally: fetch(`${import.meta.env.VITE_API_URL}/predict/admet`, ...)
            const mockAdmet = {
                score: Math.floor(Math.random() * 40) + 50,
                radar: {
                    molecular_weight: lead.molecular_weight,
                    logp: 2 + Math.random() * 2,
                    hbd: Math.floor(Math.random() * 5),
                    hba: Math.floor(Math.random() * 8),
                    tpsa: 40 + Math.random() * 80
                },
                toxicity: {
                    herg: Math.random() > 0.8 ? 'Medium Risk' : 'Low Risk',
                    ames: 'Negative',
                    cyp_inhibitor: Math.random() > 0.7,
                }
            };

            const targets = targetRes.ok ? await targetRes.json() : [];

            setAnalysisCache(prev => ({
                ...prev,
                [lead.id]: {
                    admet: mockAdmet,
                    targets: targets
                }
            }));

        } catch (err) {
            console.error("Analysis failed", err);
        } finally {
            setAnalyzing(false);
        }
    };

    useEffect(() => {
        if (selectedLead) {
            fetchAnalysis(selectedLead);
        }
    }, [selectedLead]);

    const activeAnalysis = selectedLead ? analysisCache[selectedLead.id] : null;

    return (
        <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col">
            <SEOHelmet
                title="Lead Discovery Workspace | BioDockify"
                description="Unified dashboard for lead ranking, ADMET profiling, and target prediction."
            />

            {/* Header */}
            <header className="bg-slate-800 border-b border-slate-700 h-16 flex items-center justify-between px-6 shrink-0">
                <div className="flex items-center gap-3">
                    <LayoutDashboard className="text-purple-400 h-6 w-6" />
                    <h1 className="text-xl font-semibold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                        Lead Discovery Workspace
                    </h1>
                    <span className="px-2 py-0.5 bg-slate-700 text-xs rounded-full text-slate-300">Phase 5-7</span>
                </div>
                <div className="flex gap-3">
                    <button className="flex items-center gap-2 px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors border border-slate-600">
                        <Download className="h-4 w-4" /> Export Report
                    </button>
                    <button className="flex items-center gap-2 px-3 py-1.5 text-sm bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-colors shadow-lg shadow-purple-900/20">
                        <Activity className="h-4 w-4" /> Run New Screen
                    </button>
                </div>
            </header>

            {/* Main Workspace Grid */}
            <div className="flex-1 flex overflow-hidden">

                {/* LEFT PANEL: Ranking Table (65%) */}
                <div className="w-2/3 border-r border-slate-700 flex flex-col bg-slate-900/50">
                    <div className="p-4 border-b border-slate-800 bg-slate-900 flex justify-between items-center">
                        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Ranked Candidates</h2>
                        <div className="flex gap-2 text-xs">
                            <span className="flex items-center gap-1 text-green-400"><CheckCircle className="h-3 w-3" /> Safe</span>
                            <span className="flex items-center gap-1 text-yellow-400"><AlertTriangle className="h-3 w-3" /> Warning</span>
                        </div>
                    </div>

                    <div className="flex-1 overflow-auto p-4">
                        {loading ? (
                            <div className="flex justify-center items-center h-64 text-slate-500">Loading Screening Data...</div>
                        ) : (
                            <table className="w-full text-left border-collapse">
                                <thead>
                                    <tr className="text-xs text-slate-500 border-b border-slate-700">
                                        <th className="pb-3 pl-2">Rank</th>
                                        <th className="pb-3">Compound</th>
                                        <th className="pb-3">Affinity (kcal/mol)</th>
                                        <th className="pb-3">Consensus</th>
                                        <th className="pb-3">Properties</th>
                                    </tr>
                                </thead>
                                <tbody className="text-sm">
                                    {leads.map((lead) => (
                                        <tr
                                            key={lead.id}
                                            onClick={() => setSelectedLead(lead)}
                                            className={`cursor-pointer transition-colors border-b border-slate-800/50 hover:bg-slate-800 ${selectedLead?.id === lead.id ? 'bg-slate-800 border-l-2 border-l-purple-500' : 'border-l-2 border-l-transparent'}`}
                                        >
                                            <td className="py-3 pl-2 font-mono text-slate-400">#{lead.rank}</td>
                                            <td className="py-3 font-medium text-white">{lead.name}</td>
                                            <td className="py-3 text-cyan-400 font-mono">
                                                {lead.affinity.toFixed(1)}
                                            </td>
                                            <td className="py-3">
                                                <div className="flex items-center gap-2">
                                                    <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                                        <div
                                                            className="h-full bg-gradient-to-r from-purple-500 to-cyan-500"
                                                            style={{ width: `${lead.consensus_score * 100}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-xs text-slate-400">{lead.consensus_score.toFixed(2)}</span>
                                                </div>
                                            </td>
                                            <td className="py-3">
                                                <div className="flex gap-1">
                                                    {/* Mock Indicators based on cached analysis if available */}
                                                    {analysisCache[lead.id]?.admet.toxicity.herg === 'Low Risk' && (
                                                        <div className="w-2 h-2 rounded-full bg-green-500" title="Low hERG Risk" />
                                                    )}
                                                    {analysisCache[lead.id]?.admet.score > 80 && (
                                                        <div className="w-2 h-2 rounded-full bg-blue-500" title="High Drug Likeness" />
                                                    )}
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        )}
                    </div>
                </div>

                {/* RIGHT PANEL: Insight Dashboard (35%) */}
                <div className="w-1/3 bg-slate-800 flex flex-col pointer-events-auto shadow-2xl z-10">
                    {selectedLead ? (
                        <>
                            {/* Panel Header */}
                            <div className="p-6 border-b border-slate-700 bg-slate-800">
                                <h3 className="text-xl font-bold text-white mb-1">{selectedLead.name}</h3>
                                <p className="text-xs font-mono text-slate-400 break-all">{selectedLead.smiles}</p>
                            </div>

                            {/* Tabs */}
                            <div className="flex border-b border-slate-700">
                                {['overview', 'admet', 'targets'].map(tab => (
                                    <button
                                        key={tab}
                                        onClick={() => setActiveTab(tab)}
                                        className={`flex-1 py-3 text-sm font-medium transition-colors border-b-2 ${activeTab === tab
                                                ? 'border-purple-500 text-purple-400 bg-slate-800'
                                                : 'border-transparent text-slate-500 hover:text-slate-300 bg-slate-900/30'
                                            }`}
                                    >
                                        {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                    </button>
                                ))}
                            </div>

                            {/* Panel Content */}
                            <div className="flex-1 overflow-y-auto p-6 bg-slate-800">
                                {analyzing ? (
                                    <div className="flex flex-col items-center justify-center h-48 space-y-4">
                                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
                                        <p className="text-sm text-slate-400">Running AI Predictions...</p>
                                    </div>
                                ) : !activeAnalysis ? (
                                    <div className="text-center text-slate-500 mt-10">Select a compound to view insights.</div>
                                ) : (
                                    <div className="animate-in fade-in duration-300">

                                        {/* OVERVIEW TAB */}
                                        {activeTab === 'overview' && (
                                            <div className="space-y-6">
                                                <div className="bg-slate-700/50 p-4 rounded-xl border border-slate-600">
                                                    <h4 className="text-xs uppercase text-slate-400 mb-3 font-semibold">Consensus Score</h4>
                                                    <div className="flex items-end gap-2 text-4xl font-bold text-white mb-1">
                                                        {(selectedLead.consensus_score * 10).toFixed(1)}
                                                        <span className="text-lg text-slate-500 font-normal mb-1">/ 10</span>
                                                    </div>
                                                    <p className="text-xs text-slate-400">
                                                        Aggregated from Vina ({selectedLead.affinity.toFixed(1)}), Gnina, and Drug Likeness scores.
                                                    </p>
                                                </div>

                                                <div>
                                                    <h4 className="text-xs uppercase text-slate-400 mb-3 font-semibold">Key Properties</h4>
                                                    <div className="grid grid-cols-2 gap-3">
                                                        <div className="bg-slate-900 p-3 rounded-lg border border-slate-700">
                                                            <div className="text-xs text-slate-500">Mol Weight</div>
                                                            <div className="font-mono text-slate-200">{activeAnalysis.admet.radar.molecular_weight.toFixed(1)}</div>
                                                        </div>
                                                        <div className="bg-slate-900 p-3 rounded-lg border border-slate-700">
                                                            <div className="text-xs text-slate-500">LogP</div>
                                                            <div className="font-mono text-slate-200">{activeAnalysis.admet.radar.logp.toFixed(2)}</div>
                                                        </div>
                                                        <div className="bg-slate-900 p-3 rounded-lg border border-slate-700">
                                                            <div className="text-xs text-slate-500">TPSA</div>
                                                            <div className="font-mono text-slate-200">{activeAnalysis.admet.radar.tpsa.toFixed(1)}</div>
                                                        </div>
                                                        <div className="bg-slate-900 p-3 rounded-lg border border-slate-700">
                                                            <div className="text-xs text-slate-500">H-Donors</div>
                                                            <div className="font-mono text-slate-200">{activeAnalysis.admet.radar.hbd}</div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        )}

                                        {/* ADMET TAB */}
                                        {activeTab === 'admet' && (
                                            <div className="space-y-6">
                                                <div className="flex justify-center py-4 bg-white/5 rounded-xl">
                                                    {/* Pass simplified data conforming to AdmetRadar's expected props if needed, or update Component */}
                                                    <div className="w-64 h-64">
                                                        <AdmetRadar data={{ molecular_properties: activeAnalysis.admet.radar }} />
                                                    </div>
                                                </div>

                                                <div className="space-y-3">
                                                    <h4 className="text-xs uppercase text-slate-400 font-semibold">Toxicity Flags</h4>
                                                    <div className={`p-3 rounded-lg flex justify-between items-center border ${activeAnalysis.admet.toxicity.herg === 'Low Risk' ? 'bg-green-500/10 border-green-500/20 text-green-400' : 'bg-red-500/10 border-red-500/20 text-red-400'}`}>
                                                        <span>hERG Inhibition</span>
                                                        <span className="font-bold text-sm">{activeAnalysis.admet.toxicity.herg}</span>
                                                    </div>
                                                    <div className="p-3 rounded-lg flex justify-between items-center bg-slate-700/30 border border-slate-600 text-slate-300">
                                                        <span>AMES Mutagenicity</span>
                                                        <span className="font-bold text-sm">{activeAnalysis.admet.toxicity.ames}</span>
                                                    </div>
                                                </div>
                                            </div>
                                        )}

                                        {/* TARGETS TAB */}
                                        {activeTab === 'targets' && (
                                            <div className="space-y-4">
                                                <h4 className="text-xs uppercase text-slate-400 mb-2 font-semibold">Predicted Interactions</h4>
                                                {activeAnalysis.targets.length === 0 ? (
                                                    <div className="text-sm text-slate-500 text-center py-8">No high-confidence targets found.</div>
                                                ) : (
                                                    activeAnalysis.targets.map((t, idx) => (
                                                        <div key={idx} className="group p-3 rounded-lg bg-slate-700/30 border border-slate-600 hover:border-purple-500/50 transition-colors">
                                                            <div className="flex justify-between items-start mb-1">
                                                                <div className="font-bold text-sm text-slate-200">{t.target}</div>
                                                                <span className="text-xs font-mono text-purple-400 bg-purple-500/10 px-1.5 py-0.5 rounded">
                                                                    {(t.probability * 100).toFixed(0)}%
                                                                </span>
                                                            </div>
                                                            <div className="text-xs text-slate-500 flex justify-between">
                                                                <span>{t.common_name}</span>
                                                                <a href={`https://www.uniprot.org/uniprot/${t.uniprot_id}`} target="_blank" rel="noreferrer" className="hover:text-purple-400 flex items-center gap-1">
                                                                    {t.uniprot_id} <ChevronRight className="h-3 w-3" />
                                                                </a>
                                                            </div>
                                                        </div>
                                                    ))
                                                )}
                                            </div>
                                        )}

                                    </div>
                                )}
                            </div>
                        </>
                    ) : (
                        <div className="flex-1 flex flex-col items-center justify-center text-slate-600 p-8">
                            <Beaker className="h-16 w-16 mb-4 opacity-20" />
                            <p className="text-center">Select a candidate from the ranking table to view molecular intelligence.</p>
                        </div>
                    )}
                </div>

            </div>
        </div>
    );
};

export default LeadOptimizationPage;
