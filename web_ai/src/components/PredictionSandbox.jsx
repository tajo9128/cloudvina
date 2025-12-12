import React, { useState } from 'react';
import { predictByDisease, getDiseases } from '../services/aiService';

// Disease options
const DISEASES = [
    { id: 'alzheimers', name: "Alzheimer's Disease", color: 'from-purple-500 to-indigo-600' },
    { id: 'cancer', name: 'Cancer', color: 'from-red-500 to-pink-600' },
    { id: 'diabetes', name: 'Diabetes', color: 'from-amber-500 to-orange-600' },
    { id: 'parkinson', name: "Parkinson's Disease", color: 'from-blue-500 to-cyan-600' },
    { id: 'cardiovascular', name: 'Cardiovascular', color: 'from-rose-500 to-red-600' },
];

export default function PredictionSandbox() {
    const [smiles, setSmiles] = useState('CC(C)Cc1ccc(cc1)C(C)C(=O)O');
    const [disease, setDisease] = useState('alzheimers');
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const response = await predictByDisease([smiles], disease);
            setResults(response.data.predictions);
        } catch (err) {
            console.error('Prediction error:', err);
            setError(err.response?.data?.detail || 'Prediction failed. Is the backend running?');
        } finally {
            setLoading(false);
        }
    };

    const selectedDisease = DISEASES.find(d => d.id === disease);

    return (
        <div className="bg-slate-900 rounded-2xl border border-white/10 p-8">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                <span className="text-3xl">🧬</span>
                AI Bioactivity Prediction
            </h2>

            {/* Disease Selector */}
            <div className="mb-6">
                <label className="block text-sm font-medium text-slate-400 mb-3">
                    Select Disease Target
                </label>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                    {DISEASES.map(d => (
                        <button
                            key={d.id}
                            onClick={() => setDisease(d.id)}
                            className={`p-3 rounded-xl text-sm font-medium transition-all ${disease === d.id
                                    ? `bg-gradient-to-r ${d.color} text-white shadow-lg scale-105`
                                    : 'bg-white/5 text-slate-400 hover:bg-white/10'
                                }`}
                        >
                            {d.name}
                        </button>
                    ))}
                </div>
            </div>

            {/* SMILES Input */}
            <div className="mb-6">
                <label className="block text-sm font-medium text-slate-400 mb-2">
                    Enter SMILES String
                </label>
                <div className="flex gap-3">
                    <input
                        type="text"
                        value={smiles}
                        onChange={(e) => setSmiles(e.target.value)}
                        placeholder="Enter molecule SMILES..."
                        className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-primary-500"
                    />
                    <button
                        onClick={handlePredict}
                        disabled={loading || !smiles.trim()}
                        className={`px-8 py-3 rounded-xl font-semibold transition-all ${loading
                                ? 'bg-slate-700 text-slate-400 cursor-wait'
                                : `bg-gradient-to-r ${selectedDisease?.color || 'from-primary-500 to-primary-600'} text-white hover:shadow-lg hover:scale-105`
                            }`}
                    >
                        {loading ? 'Predicting...' : 'Predict'}
                    </button>
                </div>
                <p className="text-xs text-slate-500 mt-2">
                    Example: CC(C)Cc1ccc(cc1)C(C)C(=O)O (Ibuprofen)
                </p>
            </div>

            {/* Error Display */}
            {error && (
                <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400">
                    {error}
                </div>
            )}

            {/* Results */}
            {results && results.length > 0 && (
                <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-white">Prediction Results</h3>
                    {results.map((result, idx) => (
                        <div
                            key={idx}
                            className={`p-6 rounded-xl border ${result.prediction === 'Active'
                                    ? 'bg-green-500/10 border-green-500/30'
                                    : result.prediction === 'Moderate'
                                        ? 'bg-yellow-500/10 border-yellow-500/30'
                                        : 'bg-red-500/10 border-red-500/30'
                                }`}
                        >
                            <div className="flex items-center justify-between mb-4">
                                <span className={`px-4 py-1 rounded-full text-sm font-bold ${result.prediction === 'Active'
                                        ? 'bg-green-500 text-white'
                                        : result.prediction === 'Moderate'
                                            ? 'bg-yellow-500 text-black'
                                            : 'bg-red-500 text-white'
                                    }`}>
                                    {result.prediction}
                                </span>
                                <span className="text-2xl font-bold text-white">
                                    {(result.score * 100).toFixed(1)}%
                                </span>
                            </div>

                            <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                    <span className="text-slate-400">Confidence:</span>
                                    <span className="text-white ml-2">{(result.confidence * 100).toFixed(1)}%</span>
                                </div>
                                <div>
                                    <span className="text-slate-400">Disease:</span>
                                    <span className="text-white ml-2 capitalize">{result.disease_target}</span>
                                </div>
                            </div>

                            {result.interpretation && (
                                <p className="mt-4 text-sm text-slate-300 italic">
                                    {result.interpretation}
                                </p>
                            )}

                            <p className="mt-3 text-xs text-slate-500 font-mono break-all">
                                {result.smiles}
                            </p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
