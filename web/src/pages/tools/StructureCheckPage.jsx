import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import Header from '../../components/Header';

export default function StructureCheckPage() {
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
            const response = await fetch(`${import.meta.env.VITE_API_URL}/agent/tools/rcsb`, {
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

                <h1 className="text-3xl font-bold text-slate-900 mb-2">Structure Quality Checker 🏗️</h1>
                <p className="text-slate-600 mb-8">
                    Validate PDB files before docking. Ensure high resolution (< 2.5Å) and check experimental methods.
                </p>

                <div className="bg-white rounded-2xl shadow-sm p-6 mb-8">
                    <form onSubmit={handleSearch} className="flex gap-4">
                        <input
                            type="text"
                            placeholder="Enter PDB ID (e.g. 1HSG)"
                            className="flex-1 px-4 py-3 border border-slate-200 rounded-xl focus:ring-2 focus:ring-violet-500 outline-none uppercase font-mono"
                            value={pdbId}
                            onChange={(e) => setPdbId(e.target.value)}
                            maxLength={4}
                        />
                        <button
                            type="submit"
                            disabled={loading || !pdbId}
                            className="bg-violet-600 text-white px-8 py-3 rounded-xl font-bold hover:bg-violet-700 disabled:opacity-50 transition-colors"
                        >
                            {loading ? 'Checking...' : 'Check PDB'}
                        </button>
                    </form>
                    {error && <p className="text-red-500 mt-3 font-medium">⚠️ {error}</p>}
                </div>

                {result && (
                    <div className="bg-white rounded-2xl shadow-lg p-8 border border-slate-100 animate-fade-in-up">
                        <div className="flex justify-between items-start mb-6">
                            <div>
                                <h2 className="text-xl font-bold text-slate-900 leading-tight">{result.title}</h2>
                                <p className="text-violet-600 font-mono text-sm mt-2">PDB: {result.pdb_id} • Released: {result.release_date}</p>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-8">
                            <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center">
                                <p className="text-xs font-bold text-slate-500 uppercase">Method</p>
                                <p className="text-lg font-bold text-slate-900">{result.method}</p>
                            </div>
                            <div className={`p-4 rounded-xl border text-center ${parseFloat(result.resolution) < 2.5 ? 'bg-green-50 border-green-100' : 'bg-yellow-50 border-yellow-100'}`}>
                                <p className="text-xs font-bold text-slate-500 uppercase">Resolution</p>
                                <p className={`text-lg font-bold ${parseFloat(result.resolution) < 2.5 ? 'text-green-700' : 'text-yellow-700'}`}>
                                    {result.resolution} Å
                                </p>
                            </div>
                            <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center">
                                <p className="text-xs font-bold text-slate-500 uppercase">Organism</p>
                                <p className="text-lg font-bold text-slate-900 truncate">{result.organism}</p>
                            </div>
                        </div>

                        {/* Quality Badge */}
                        <div className={`text-center p-4 rounded-xl border-dashed border-2 ${parseFloat(result.resolution) < 2.5 ? 'border-green-200 bg-green-50/50' : 'border-red-200 bg-red-50/50'}`}>
                            {parseFloat(result.resolution) < 2.5 ? (
                                <p className="text-green-700 font-bold">✅ High Quality Structure - Recommended for Docking</p>
                            ) : (
                                <p className="text-red-600 font-bold">⚠️ Low Resolution ({'>'} 2.5Å) - Docking results may be less accurate</p>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
