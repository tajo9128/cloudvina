import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import Header from '../../components/Header';

export default function DevelopabilityPage() {
    const [smiles, setSmiles] = useState('CC(=O)OC1=CC=CC=C1C(=O)O'); // Aspirin
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleCheck = async () => {
        if (!smiles.trim()) return;
        setLoading(true);
        setError('');
        setResult(null);

        try {
            const response = await fetch(`${import.meta.env.VITE_API_URL}/agent/tools/developability`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ smiles })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || data.error || 'Failed to fetch');
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

                <h1 className="text-3xl font-bold text-slate-900 mb-2">Developability Warning System ⚠️</h1>
                <p className="text-slate-600 mb-8 max-w-2xl">
                    Don't waste time on undevelopable compounds. Check for PAINS, structural alerts, and violations of drug-likeness rules before you start docking.
                </p>

                <div className="bg-white rounded-2xl shadow-sm p-8 mb-8 border border-slate-100">
                    <label className="block text-sm font-bold text-slate-700 mb-3 uppercase tracking-wider">Input SMILES String</label>
                    <div className="flex gap-4">
                        <input
                            type="text"
                            value={smiles}
                            onChange={(e) => setSmiles(e.target.value)}
                            className="flex-1 px-4 py-3 border border-slate-200 rounded-xl focus:ring-2 focus:ring-amber-500 outline-none font-mono text-sm"
                            placeholder="Enter SMILES..."
                        />
                        <button
                            onClick={handleCheck}
                            disabled={loading || !smiles}
                            className="bg-amber-500 text-white px-8 py-3 rounded-xl font-bold hover:bg-amber-600 disabled:opacity-50 transition-colors shadow-lg shadow-amber-200"
                        >
                            {loading ? 'Analyzing...' : 'Run Safety Check'}
                        </button>
                    </div>
                    {error && <p className="text-red-500 text-sm mt-3 bg-red-50 p-2 rounded border border-red-100">{error}</p>}
                </div>

                {result && (
                    <div className="animate-fade-in-up">
                        {/* HEADER CARD */}
                        <div className={`rounded-2xl p-6 mb-6 shadow-md border-l-8 flex items-center justify-between ${result.risk_level === 'Low' ? 'bg-white border-green-500' :
                            result.risk_level === 'Medium' ? 'bg-white border-yellow-500' :
                                'bg-white border-red-500'
                            }`}>
                            <div>
                                <h2 className="text-2xl font-bold text-slate-800 border-none">Risk Level: <span className={
                                    result.risk_level === 'Low' ? 'text-green-600' :
                                        result.risk_level === 'Medium' ? 'text-yellow-600' :
                                            'text-red-600'
                                }>{result.risk_level}</span></h2>
                                <p className="text-slate-500 mt-1">{result.conclusion}</p>
                            </div>
                            <div className="text-5xl">
                                {result.risk_level === 'Low' ? '✅' :
                                    result.risk_level === 'Medium' ? '⚠️' : '🚫'}
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* PROPERTIES CARD */}
                            <div className="bg-white rounded-2xl shadow-sm p-6">
                                <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                                    🧪 Molecular Properties
                                </h3>
                                <div className="space-y-3">
                                    <PropRow label="Molecular Weight" value={result.properties.mw} limit="&lt; 500" />
                                    <PropRow label="LogP (Lipophilicity)" value={result.properties.logp} limit="&lt; 5" />
                                    <PropRow label="H-Bond Donors" value={result.properties.hbd} limit="&lt; 5" />
                                    <PropRow label="H-Bond Acceptors" value={result.properties.hba} limit="&lt; 10" />
                                    <PropRow label="TPSA" value={result.properties.tpsa} limit="&lt; 140" />
                                    <PropRow label="Rotatable Bonds" value={result.properties.rotatable_bonds} limit="&lt; 10" />
                                </div>
                            </div>

                            {/* ALERTS CARD */}
                            <div className="bg-white rounded-2xl shadow-sm p-6">
                                <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                                    🚨 Structural Alerts
                                </h3>
                                {result.structural_alerts.length === 0 && result.lipinski_violations.length === 0 ? (
                                    <div className="h-full flex flex-col items-center justify-center text-slate-400 pb-8">
                                        <div className="text-4xl mb-2">🎉</div>
                                        <p>No structural alerts found.</p>
                                    </div>
                                ) : (
                                    <div className="space-y-4">
                                        {result.lipinski_violations.map((v, i) => (
                                            <div key={i} className="bg-yellow-50 text-yellow-800 p-3 rounded-lg text-sm border border-yellow-100 flex items-start gap-2">
                                                <span>⚠️</span>
                                                <span className="font-semibold">{v}</span>
                                            </div>
                                        ))}
                                        {result.structural_alerts.map((a, i) => (
                                            <div key={i} className="bg-red-50 text-red-800 p-3 rounded-lg text-sm border border-red-100 flex items-start gap-2">
                                                <span>🚫</span>
                                                <span className="font-semibold">{a}</span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Agent Zero Integration */}
                        <div className="mt-8 text-center">
                            <button
                                onClick={() => window.dispatchEvent(new CustomEvent('agent-zero-trigger', {
                                    detail: { prompt: `Analyze this developability report for SMILES ${result.smiles}: ${JSON.stringify(result)}. Should I proceed with docking?`, context: result }
                                }))}
                                className="text-indigo-600 font-bold hover:text-indigo-800 transition-colors"
                            >
                                Ask Agent Zero for a Second Opinion 🤖
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

function PropRow({ label, value, limit }) {
    return (
        <div className="flex justify-between items-center border-b border-slate-50 last:border-0 pb-2 last:pb-0">
            <span className="text-slate-600 text-sm">{label} <span className="text-xs text-slate-400">({limit})</span></span>
            <span className="font-mono font-bold text-slate-800">{value}</span>
        </div>
    )
}
