import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import Header from '../../components/Header';

export default function BioactivityPage() {
    const [chemblId, setChemblId] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSearch = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setResult(null);

        try {
            const response = await fetch(`${import.meta.env.VITE_API_URL}/agent/tools/chembl`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: chemblId })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Failed to fetch');
            if (data.error) throw new Error(data.error);

            setResult(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50">
            <Header />
            <div className="pt-24 pb-12 container mx-auto px-4 max-w-4xl">
                <Link to="/tools" className="text-slate-500 hover:text-slate-800 mb-6 inline-block">← Back to Tools</Link>

                <h1 className="text-3xl font-bold text-slate-900 mb-2">ChEMBL Bioactivity Lookup 💊</h1>
                <p className="text-slate-600 mb-8">
                    Query experimental data (IC50, Ki, Kd) for known inhibitors. Benchmark your docking scores against reality.
                </p>

                <div className="bg-white rounded-2xl shadow-sm p-6 mb-8">
                    <form onSubmit={handleSearch} className="flex gap-4">
                        <input
                            type="text"
                            placeholder="Enter ChEMBL ID (e.g. CHEMBL25)"
                            className="flex-1 px-4 py-3 border border-slate-200 rounded-xl focus:ring-2 focus:ring-rose-500 outline-none uppercase font-mono"
                            value={chemblId}
                            onChange={(e) => setChemblId(e.target.value)}
                        />
                        <button
                            type="submit"
                            disabled={loading || !chemblId}
                            className="bg-rose-600 text-white px-8 py-3 rounded-xl font-bold hover:bg-rose-700 disabled:opacity-50 transition-colors"
                        >
                            {loading ? 'Searching...' : 'Search Activities'}
                        </button>
                    </form>
                    {error && <p className="text-red-500 mt-3 font-medium">⚠️ {error}</p>}
                </div>

                {result && (
                    <div className="bg-white rounded-2xl shadow-lg p-8 border border-slate-100 animate-fade-in-up">
                        <div className="flex justify-between items-center mb-6">
                            <h2 className="text-xl font-bold text-slate-900">Activity Data for {result.chembl_id}</h2>
                            <span className="bg-rose-50 text-rose-600 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide">
                                {result.found_activities} Records Found
                            </span>
                        </div>

                        {result.top_activities && result.top_activities.length > 0 ? (
                            <div className="overflow-x-auto">
                                <table className="w-full text-left border-collapse">
                                    <thead>
                                        <tr className="border-b border-rose-100 text-rose-800 bg-rose-50">
                                            <th className="p-3 font-bold text-sm">Type</th>
                                            <th className="p-3 font-bold text-sm">Value</th>
                                            <th className="p-3 font-bold text-sm">Target</th>
                                            <th className="p-3 font-bold text-sm">Source</th>
                                        </tr>
                                    </thead>
                                    <tbody className="text-slate-700">
                                        {result.top_activities.map((act, i) => {
                                            // Crude manual parsing of the string returned by tool (Tool returns formatted strings)
                                            // "IC50 = 10.5 nM (Target: CHEMBL123)"
                                            // Ideally tool returns dicts, but we kept it simple for Agent usage.
                                            // Let's just render the string as a list item for stability.
                                            return null;
                                        })}
                                        {/* Since tool returns strings, let's just list them nicely */}
                                        {result.top_activities.map((actString, i) => (
                                            <tr key={i} className="border-b border-slate-50 hover:bg-slate-50 transition-colors">
                                                <td className="p-4 font-mono text-sm" colSpan="4">
                                                    {actString}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : (
                            <p className="text-slate-500 italic">No bioactivity data found.</p>
                        )}

                        <div className="mt-8 pt-6 border-t border-slate-100 flex justify-end">
                            <button
                                onClick={() => window.dispatchEvent(new CustomEvent('agent-zero-trigger', {
                                    detail: { prompt: `I found these activities for ${result.chembl_id}: ${JSON.stringify(result.top_activities)}. Is this a potent inhibitor?`, context: result }
                                }))}
                                className="text-rose-600 font-bold hover:underline flex items-center gap-2"
                            >
                                Compare with BioDockify AI Agent 🤖
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
