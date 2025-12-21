import React, { useState } from 'react';
import { supabase } from '../supabaseClient';
import { Link } from 'react-router-dom';

const TargetPredictionPage = () => {
    const [smiles, setSmiles] = useState('');
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    const handlePredict = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const { data: { session } } = await supabase.auth.getSession();

            // Call our new API endpoint
            const response = await fetch(`${import.meta.env.VITE_API_URL}/predict/target/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session?.access_token}`
                },
                body: JSON.stringify({ smiles })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Prediction failed');
            }

            const data = await response.json();
            setResults(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-[#0B1121] to-slate-900 text-white pt-24 px-4 pb-12">
            <div className="container mx-auto max-w-4xl">
                <div className="flex items-center gap-4 mb-8">
                    <Link to="/tools/converter" className="text-slate-400 hover:text-white transition-colors">
                        ← Back to Tools
                    </Link>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                        Target Prediction
                    </h1>
                </div>

                <div className="bg-slate-800/50 backdrop-blur-md rounded-2xl border border-slate-700 p-8 shadow-xl">
                    <p className="text-slate-300 mb-6">
                        Predict potential protein targets for your small molecule.
                        Enter a SMILES string below.
                    </p>

                    <form onSubmit={handlePredict} className="space-y-6">
                        <div>
                            <label className="block text-sm font-medium text-slate-400 mb-2">
                                SMILES String
                            </label>
                            <input
                                type="text"
                                value={smiles}
                                onChange={(e) => setSmiles(e.target.value)}
                                placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
                                className="w-full bg-slate-900/50 border border-slate-600 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all font-mono"
                                required
                            />
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className={`w-full py-4 rounded-xl font-bold text-lg shadow-lg transition-all transform hover:-translate-y-0.5 ${loading
                                ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                                : 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:shadow-blue-500/25'
                                }`}
                        >
                            {loading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                    </svg>
                                    Predicting Targets...
                                </span>
                            ) : (
                                'Run Prediction 🎯'
                            )}
                        </button>
                    </form>

                    {error && (
                        <div className="mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-center">
                            {error}
                        </div>
                    )}
                </div>

                {results && (
                    <div className="mt-8 space-y-6 animate-fade-in">
                        <h2 className="text-xl font-bold text-white mb-4">Predicted API Targets</h2>
                        <div className="grid gap-4">
                            {results.length === 0 ? (
                                <div className="p-8 bg-slate-800/30 rounded-xl text-center text-slate-400">
                                    No reliable targets found for this molecule.
                                </div>
                            ) : (
                                results.map((target, idx) => (
                                    <div key={idx} className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-6 hover:border-blue-500/30 transition-colors flex items-center justify-between">
                                        <div>
                                            <h3 className="text-lg font-bold text-white">{target.target}</h3>
                                            <p className="text-slate-400 text-sm">
                                                Organism: {target.common_name} • ID: {' '}
                                                <a
                                                    href={target.uniprot_id.startsWith('CHEMBL')
                                                        ? `https://www.ebi.ac.uk/chembl/target_report_card/${target.uniprot_id}`
                                                        : `https://www.uniprot.org/uniprot/${target.uniprot_id}`}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    className="text-blue-400 hover:underline"
                                                >
                                                    {target.uniprot_id}
                                                </a>
                                            </p>
                                        </div>
                                        <div className="flex flex-col items-end">
                                            <div className="text-2xl font-bold text-green-400">
                                                {(target.probability * 100).toFixed(0)}%
                                            </div>
                                            <span className="text-xs text-slate-500 uppercase tracking-wider">Probability</span>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default TargetPredictionPage;
