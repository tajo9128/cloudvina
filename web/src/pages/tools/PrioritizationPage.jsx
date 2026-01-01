import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import Header from '../../components/Header';

export default function PrioritizationPage() {
    const [leadsText, setLeadsText] = useState(
        `[
  { "id": "Compound_A", "affinity": -10.2, "cost": 500, "mw": 320, "logp": 2.5 },
  { "id": "Compound_B", "affinity": -9.8, "cost": 1200, "mw": 650, "logp": 5.2 },
  { "id": "Compound_C", "affinity": -8.5, "cost": 100, "mw": 200, "logp": 1.2 },
  { "id": "Compound_D", "affinity": -11.5, "cost": 3000, "mw": 450, "logp": 3.8 },
  { "id": "Compound_E", "affinity": -7.0, "cost": 50, "mw": 180, "logp": 0.8 }
]`
    );
    const [budget, setBudget] = useState(2500);
    const [risk, setRisk] = useState('medium');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleOptimize = async () => {
        setLoading(true);
        setError('');
        setResult(null);

        try {
            let leads;
            try {
                leads = JSON.parse(leadsText);
            } catch (e) {
                throw new Error("Invalid JSON format. Please check your syntax.");
            }

            const response = await fetch(`${import.meta.env.VITE_API_URL}/agent/tools/prioritization`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ leads, budget, risk })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Failed to fetch');

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
            <div className="pt-24 pb-12 container mx-auto px-4 max-w-6xl">
                <Link to="/tools" className="text-slate-500 hover:text-slate-800 mb-6 inline-block">← Back to Tools</Link>

                <h1 className="text-3xl font-bold text-slate-900 mb-2">Lead Portfolio Optimizer ⚖️</h1>
                <p className="text-slate-600 mb-8 max-w-2xl">
                    Apply decision science to your drug discovery pipeline.
                    Input your candidates, budget, and risk tolerance, and let our engine determine the optimal testing strategy using Knapsack heuristics.
                </p>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* INPUT COLUMN */}
                    <div className="space-y-6">
                        <div className="bg-white rounded-2xl shadow-sm p-6">
                            <h2 className="text-lg font-bold text-slate-800 mb-4">1. Define Constraints</h2>

                            <div className="mb-6">
                                <label className="block text-sm font-bold text-slate-700 mb-2">Total Budget ($)</label>
                                <input
                                    type="number"
                                    value={budget}
                                    onChange={(e) => setBudget(parseFloat(e.target.value))}
                                    className="w-full px-4 py-2 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 outline-none font-mono"
                                />
                                <div className="mt-2">
                                    <input
                                        type="range"
                                        min="100" max="10000" step="100"
                                        value={budget}
                                        onChange={(e) => setBudget(parseFloat(e.target.value))}
                                        className="w-full accent-indigo-600"
                                    />
                                </div>
                            </div>

                            <div className="mb-6">
                                <label className="block text-sm font-bold text-slate-700 mb-2">Risk Tolerance</label>
                                <div className="grid grid-cols-3 gap-2">
                                    {['low', 'medium', 'high'].map((r) => (
                                        <button
                                            key={r}
                                            onClick={() => setRisk(r)}
                                            className={`py-2 rounded-lg text-sm font-bold capitalize transition-colors ${risk === r
                                                    ? 'bg-indigo-600 text-white'
                                                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                                                }`}
                                        >
                                            {r}
                                        </button>
                                    ))}
                                </div>
                                <p className="text-xs text-slate-500 mt-2">
                                    {risk === 'high' ? 'Maximize raw affinity, ignore properties.' :
                                        risk === 'low' ? 'Prioritize drug-likeness (QED/Lipinski) first.' :
                                            'Balanced approach.'}
                                </p>
                            </div>
                        </div>

                        <div className="bg-white rounded-2xl shadow-sm p-6 flex flex-col h-[400px]">
                            <h2 className="text-lg font-bold text-slate-800 mb-4">2. Input Candidates (JSON)</h2>
                            <textarea
                                className="flex-1 w-full p-4 bg-slate-50 border border-slate-200 rounded-xl font-mono text-xs focus:ring-2 focus:ring-indigo-500 outline-none resize-none"
                                value={leadsText}
                                onChange={(e) => setLeadsText(e.target.value)}
                            />
                            <button
                                onClick={handleOptimize}
                                disabled={loading}
                                className="mt-4 w-full bg-indigo-600 text-white py-3 rounded-xl font-bold hover:bg-indigo-700 disabled:opacity-50 transition-colors shadow-lg shadow-indigo-200"
                            >
                                {loading ? 'Optimizing...' : 'Run Optimization Strategy 🚀'}
                            </button>
                            {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
                        </div>
                    </div>

                    {/* RESULTS COLUMN */}
                    <div className="lg:h-full">
                        {result ? (
                            <div className="bg-white rounded-2xl shadow-lg border border-slate-200 overflow-hidden h-full flex flex-col animate-fade-in-up">
                                <div className="bg-slate-900 text-white p-6">
                                    <h3 className="text-lg font-bold flex items-center gap-2">
                                        Strategy Report: {result.strategy}
                                    </h3>
                                    <div className="mt-4 flex gap-6">
                                        <div>
                                            <div className="text-xs text-slate-400 uppercase">Selected</div>
                                            <div className="text-2xl font-bold text-emerald-400">{result.selected_count}</div>
                                        </div>
                                        <div>
                                            <div className="text-xs text-slate-400 uppercase">Budget Used</div>
                                            <div className="text-2xl font-bold text-white">${result.budget_used}</div>
                                        </div>
                                        <div>
                                            <div className="text-xs text-slate-400 uppercase">Remaining</div>
                                            <div className="text-2xl font-bold text-slate-400">${result.remaining_budget}</div>
                                        </div>
                                    </div>
                                </div>

                                <div className="p-0 overflow-y-auto flex-1 bg-slate-50">
                                    {result.selected.map((lead, i) => (
                                        <div key={i} className="bg-white p-4 border-b border-slate-100 flex justify-between items-center hover:bg-emerald-50/30 transition-colors border-l-4 border-l-emerald-500">
                                            <div>
                                                <div className="font-bold text-slate-900">{lead.id}</div>
                                                <div className="text-xs text-slate-500">Affinity: {lead.affinity} | Cost: ${lead.normalized_cost}</div>
                                            </div>
                                            <div className="text-right">
                                                <span className="bg-emerald-100 text-emerald-700 px-2 py-1 rounded text-xs font-bold">SELECTED</span>
                                            </div>
                                        </div>
                                    ))}

                                    {result.discarded.map((lead, i) => (
                                        <div key={i} className="bg-slate-50 p-4 border-b border-slate-200 flex justify-between items-center opacity-75 grayscale hover:grayscale-0 transition-all">
                                            <div>
                                                <div className="font-bold text-slate-700">{lead.id}</div>
                                                <div className="text-xs text-slate-500">Affinity: {lead.affinity} | Cost: ${lead.normalized_cost}</div>
                                            </div>
                                            <div className="text-right">
                                                <div className="text-xs font-bold text-red-500">{lead.reason}</div>
                                                <span className="text-[10px] text-slate-400 uppercase">Discarded</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>

                                <div className="p-4 bg-white border-t border-slate-200">
                                    <button
                                        onClick={() => window.dispatchEvent(new CustomEvent('agent-zero-trigger', {
                                            detail: { prompt: `Analyze this portfolio strategy: ${JSON.stringify(result)}. Is it too risky?`, context: result }
                                        }))}
                                        className="w-full text-indigo-600 font-bold hover:bg-indigo-50 py-3 rounded-xl transition-colors"
                                    >
                                        Request Agent Zero Audit 🤖
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <div className="h-full bg-slate-100 rounded-2xl border-2 border-dashed border-slate-300 flex items-center justify-center text-slate-400 p-8 text-center">
                                <div>
                                    <div className="text-4xl mb-4">📊</div>
                                    <p>Enter constraints and leads to generate an optimized portfolio.</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
