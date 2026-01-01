import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import Header from '../../components/Header';

export default function PocketDetectPage() {
    const [pdbId, setPdbId] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSearch = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setResult(null);

        try {
            const response = await fetch(`${import.meta.env.VITE_API_URL}/agent/tools/pockets`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: pdbId })
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

                <h1 className="text-3xl font-bold text-slate-900 mb-2">Binding Pocket Detector 🎯</h1>
                <p className="text-slate-600 mb-8">
                    Identify potential binding sites on a protein. Returns optimal grid box coordinates for docking.
                </p>

                <div className="bg-white rounded-2xl shadow-sm p-6 mb-8">
                    <form onSubmit={handleSearch} className="flex gap-4">
                        <input
                            type="text"
                            placeholder="Enter PDB ID (e.g. 1HSG)"
                            className="flex-1 px-4 py-3 border border-slate-200 rounded-xl focus:ring-2 focus:ring-orange-500 outline-none uppercase font-mono"
                            value={pdbId}
                            onChange={(e) => setPdbId(e.target.value)}
                            maxLength={4}
                        />
                        <button
                            type="submit"
                            disabled={loading || !pdbId}
                            className="bg-orange-600 text-white px-8 py-3 rounded-xl font-bold hover:bg-orange-700 disabled:opacity-50 transition-colors"
                        >
                            {loading ? 'Detect Pockets' : 'Find Sites'}
                        </button>
                    </form>
                    {error && <p className="text-red-500 mt-3 font-medium">⚠️ {error}</p>}
                </div>

                {result && (
                    <div className="bg-white rounded-2xl shadow-lg p-8 border border-slate-100 animate-fade-in-up">
                        <div className="mb-6">
                            <h2 className="text-xl font-bold text-slate-900">Analysis for {result.pdb_id}</h2>
                            <p className="text-orange-600 font-medium">{result.recommendation}</p>
                        </div>

                        {result.top_pockets && result.top_pockets.length > 0 && (
                            <div className="grid grid-cols-1 gap-4">
                                {result.top_pockets.map((pocket, i) => (
                                    <div key={i} className="bg-slate-50 p-5 rounded-xl border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
                                        <div className="flex items-center gap-4">
                                            <div className="w-10 h-10 rounded-full bg-orange-100 text-orange-600 flex items-center justify-center font-bold text-lg">
                                                {pocket.rank}
                                            </div>
                                            <div>
                                                <p className="text-xs font-bold text-slate-500 uppercase tracking-wide">CONFIDENCE SCORE</p>
                                                <p className="text-lg font-bold text-slate-900">{pocket.score ? parseFloat(pocket.score).toFixed(2) : 'N/A'}</p>
                                            </div>
                                        </div>

                                        <div className="text-right">
                                            <p className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-1">Center Coordinates (X, Y, Z)</p>
                                            <code className="bg-white px-3 py-1 rounded border border-slate-200 font-mono text-sm text-slate-700 block">
                                                {pocket.center ? `[${pocket.center.join(', ')}]` : 'N/A'}
                                            </code>
                                        </div>

                                        <div className="text-right">
                                            <p className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-1">Box Size</p>
                                            <code className="bg-white px-3 py-1 rounded border border-slate-200 font-mono text-sm text-slate-700 block">
                                                {pocket.size ? `[${pocket.size.join(', ')}]` : 'N/A'}
                                            </code>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        <div className="mt-8 pt-6 border-t border-slate-100 flex justify-end">
                            <button
                                onClick={() => window.dispatchEvent(new CustomEvent('agent-zero-trigger', {
                                    detail: { prompt: `I found these pockets for ${result.pdb_id}. Which one looks most promising for drug discovery?`, context: result }
                                }))}
                                className="text-orange-600 font-bold hover:underline flex items-center gap-2"
                            >
                                Analyze via Agent Zero 🤖
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
