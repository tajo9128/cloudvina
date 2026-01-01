import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import Header from '../components/Header';

export default function TargetExplorerPage() {
    const [targetId, setTargetId] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSearch = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setResult(null);

        try {
            const response = await fetch(`${import.meta.env.VITE_API_URL}/agent/tools/uniprot`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: targetId })
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

                <h1 className="text-3xl font-bold text-slate-900 mb-2">UniProt Target Explorer 🧬</h1>
                <p className="text-slate-600 mb-8">
                    Instantly fetch gene names, function summaries, and active site locations for any protein target.
                </p>

                <div className="bg-white rounded-2xl shadow-sm p-6 mb-8">
                    <form onSubmit={handleSearch} className="flex gap-4">
                        <input
                            type="text"
                            placeholder="Enter UniProt ID (e.g., P53_HUMAN, P04637)"
                            className="flex-1 px-4 py-3 border border-slate-200 rounded-xl focus:ring-2 focus:ring-emerald-500 outline-none"
                            value={targetId}
                            onChange={(e) => setTargetId(e.target.value)}
                        />
                        <button
                            type="submit"
                            disabled={loading || !targetId}
                            className="bg-emerald-600 text-white px-8 py-3 rounded-xl font-bold hover:bg-emerald-700 disabled:opacity-50 transition-colors"
                        >
                            {loading ? 'Fetching...' : 'Explore Target'}
                        </button>
                    </form>
                    {error && <p className="text-red-500 mt-3 font-medium">⚠️ {error}</p>}
                </div>

                {result && (
                    <div className="bg-white rounded-2xl shadow-lg p-8 border border-slate-100 animate-fade-in-up">
                        <div className="flex justify-between items-start mb-6">
                            <div>
                                <h2 className="text-2xl font-bold text-slate-900">{result.protein_name}</h2>
                                <p className="text-emerald-600 font-mono text-sm mt-1">ID: {result.uniprot_id} • Gene: {result.gene_name}</p>
                            </div>
                            <span className="bg-slate-100 text-slate-600 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide">
                                {result.organism}
                            </span>
                        </div>

                        <div className="prose prose-slate max-w-none mb-8">
                            <h3 className="text-lg font-bold text-slate-800">Function</h3>
                            <p className="text-slate-600 leading-relaxed bg-slate-50 p-4 rounded-lg border border-slate-100">
                                {result.function}
                            </p>
                        </div>

                        {result.important_sites && result.important_sites.length > 0 && (
                            <div>
                                <h3 className="text-lg font-bold text-slate-800 mb-3">Key Features & Active Sites</h3>
                                <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    {result.important_sites.map((site, i) => (
                                        <li key={i} className="flex items-center gap-2 text-sm text-slate-700 bg-emerald-50 px-3 py-2 rounded-lg border border-emerald-100">
                                            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                                            {site}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        <div className="mt-8 pt-6 border-t border-slate-100 flex justify-end">
                            <button
                                onClick={() => window.dispatchEvent(new CustomEvent('agent-zero-trigger', {
                                    detail: { prompt: `I found target ${result.uniprot_id} (${result.gene_name}). Can you explain its function in simple terms and suggest similar targets?`, context: result }
                                }))}
                                className="text-emerald-600 font-bold hover:underline flex items-center gap-2"
                            >
                                Ask Agent Zero about this Result 🤖
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
