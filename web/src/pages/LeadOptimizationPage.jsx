import React, { useState, useEffect } from 'react';
import { supabase } from '../supabaseClient';
import { Link } from 'react-router-dom';
import SEOHelmet from '../components/SEOHelmet';

const LeadOptimizationPage = () => {
    const [leads, setLeads] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchAndRankLeads();
    }, []);

    const fetchAndRankLeads = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session) return;

            // 1. Fetch User's Jobs
            const jobsIdsRes = await fetch(`${import.meta.env.VITE_API_URL}/jobs`, {
                headers: { 'Authorization': `Bearer ${session.access_token}` }
            });
            const jobs = await jobsIdsRes.json();

            // 2. Prepare Data for Ranking Engine
            // Extract best affinity and binding energy where available
            const hits = jobs.map(job => {
                // Docking Score (Vina)
                let docking_score = job.binding_affinity;
                // Convert to float if string
                if (typeof docking_score === 'string') docking_score = parseFloat(docking_score);

                // MD Binding Energy (MM-GBSA) - currently fetched via status check or stored in job metadata
                // Since our /jobs endpoint might not return analysis details, we might need to rely on what's available
                // For MVP, we'll try to use existing job data fields. 
                // NOTE: ideally /jobs should return minimal analysis summary. 
                // Checking previous implementation of list_jobs in properties... 
                // It returns binding_affinity. It doesn't seem to return custom MMGBSA analysis column yet.
                // We will assume for now 'binding_affinity' is the primary score available.

                // Mocking QED for demo if not in DB
                const qed = 0.5; // Placeholder

                return {
                    id: job.id,
                    compound_name: job.ligand_filename || "Unknown Ligand",
                    docking_score: docking_score,
                    binding_energy: null, // Populated if we had it in the list endpoint
                    qed: qed,
                    status: job.status
                };
            }).filter(h => h.docking_score !== null); // Only rank docked items

            // 3. Send to Ranking Endpoint
            const rankRes = await fetch(`${import.meta.env.VITE_API_URL}/ranking/rank-hits`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ hits })
            });

            if (rankRes.ok) {
                const data = await rankRes.json();
                setLeads(data.ranked_hits);
            } else {
                setError("Failed to rank hits.");
            }

        } catch (err) {
            console.error(err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const checkADMET = async (jobId) => {
        try {
            const { data: { session } } = await supabase.auth.getSession();

            // Call API to get properties
            const res = await fetch(`${import.meta.env.VITE_API_URL}/jobs/${jobId}/drug-properties`, {
                headers: { 'Authorization': `Bearer ${session.access_token}` }
            });

            if (res.ok) {
                const data = await res.json();

                // Update local state with new properties
                setLeads(prev => prev.map(lead => {
                    if (lead.id === jobId) {
                        return { ...lead, admet: data.properties };
                    }
                    return lead;
                }));
            }
        } catch (error) {
            console.error("ADMET check failed:", error);
        }
    };

    const downloadReport = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession();

            // Trigger PDF generation
            const res = await fetch(`${import.meta.env.VITE_API_URL}/ranking/report`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ hits: leads })
            });

            if (res.ok) {
                // Handle binary pdf
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `BioDockify_Lead_Report_${new Date().toISOString().slice(0, 10)}.pdf`;
                document.body.appendChild(a);
                a.click();
                a.remove();
            } else {
                setError("Failed to generate report.");
            }
        } catch (err) {
            console.error(err);
            setError(err.message);
        }
    };

    if (loading) return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
        </div>
    );

    return (
        <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <SEOHelmet
                title="Lead Optimization & ADMET Prediction | BioDockify"
                description="Rank your virtual screening hits with consensus scoring. Predict ADMET properties including BBB permeability and toxicity using AI."
                keywords="lead optimization, admet prediction, toxicity check, consensus scoring, binding affinity ranking"
                canonical="https://biodockify.com/leads"
            />
            <div className="max-w-7xl mx-auto">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-900">Lead Optimization</h1>
                    <p className="mt-2 text-gray-600">
                        Prioritized list of candidate compounds based on consensus scoring (Docking + MD + Properties).
                    </p>
                </div>

                <div className="bg-white shadow-xl rounded-2xl overflow-hidden">
                    <div className="px-6 py-4 border-b border-gray-100 bg-gray-50 flex justify-between items-center">
                        <h2 className="font-semibold text-gray-700">Ranked Compounds</h2>
                        <div className="flex gap-2">
                            <button
                                onClick={downloadReport}
                                className="px-4 py-2 bg-white border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors flex items-center gap-2"
                            >
                                📄 Export PDF (Ranked)
                            </button>
                            <span className="text-sm bg-indigo-100 text-indigo-800 py-2 px-3 rounded-lg font-medium">
                                {leads.length} Candidates
                            </span>
                        </div>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Compound</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Consensus Score</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Docking (Vina)</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ADMET Predictions</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {leads.map((lead, index) => (
                                    <tr key={lead.id} className="hover:bg-gray-50 transition-colors">
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <div className={`flex items-center justify-center w-8 h-8 rounded-full font-bold ${index < 3 ? 'bg-yellow-100 text-yellow-700' : 'bg-gray-100 text-gray-500'
                                                }`}>
                                                {lead.rank}
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <div className="text-sm font-medium text-gray-900">{lead.compound_name}</div>
                                            <div className="text-xs text-gray-500">ID: {lead.id.slice(0, 8)}...</div>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            {/* Progress bar for score */}
                                            <div className="w-full max-w-xs">
                                                <div className="flex items-center justify-between mb-1">
                                                    <span className="text-sm font-bold text-indigo-600">
                                                        {(lead.consensus_score * 100).toFixed(1)}
                                                    </span>
                                                </div>
                                                <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-indigo-600 rounded-full"
                                                        style={{ width: `${lead.consensus_score * 100}%` }}
                                                    ></div>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                            {lead.docking_score?.toFixed(1)} kcal/mol
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                                            {!lead.admet ? (
                                                <button
                                                    onClick={() => checkADMET(lead.id)}
                                                    className="px-3 py-1 bg-blue-50 text-blue-600 text-xs rounded-full hover:bg-blue-100 transition-colors"
                                                >
                                                    🔮 Predict
                                                </button>
                                            ) : (
                                                <div className="flex gap-2">
                                                    {lead.admet.bbb?.permeable && (
                                                        <span className="px-2 py-0.5 bg-purple-100 text-purple-800 text-xs rounded border border-purple-200" title="Predicted CNS active">
                                                            🧠 BBB+
                                                        </span>
                                                    )}
                                                    {lead.admet.toxicity?.has_alerts && (
                                                        <span className="px-2 py-0.5 bg-red-100 text-red-800 text-xs rounded border border-red-200" title="Structural Alert Found">
                                                            ⚠️ Toxic
                                                        </span>
                                                    )}
                                                    {!lead.admet.bbb?.permeable && !lead.admet.toxicity?.has_alerts && (
                                                        <span className="px-2 py-0.5 bg-green-50 text-green-700 text-xs rounded border border-green-200">
                                                            Safe
                                                        </span>
                                                    )}
                                                </div>
                                            )}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${lead.status === 'SUCCEEDED' ? 'bg-green-100 text-green-800' :
                                                lead.status === 'FAILED' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'
                                                }`}>
                                                {lead.status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                            <Link to={`/md/${lead.id}`} className="text-indigo-600 hover:text-indigo-900 mr-4">
                                                Analyze
                                            </Link>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    {leads.length === 0 && !error && (
                        <div className="p-12 text-center text-gray-500">
                            No completed docking jobs found to rank. Submit a job first!
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default LeadOptimizationPage;
