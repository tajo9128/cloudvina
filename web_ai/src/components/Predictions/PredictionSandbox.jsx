import React, { useState } from 'react';
import { predictByDisease } from '../../services/aiService';

// Disease options
const DISEASES = [
    { id: 'alzheimers', name: "Alzheimer's", color: 'from-purple-500 to-indigo-600' },
    { id: 'cancer', name: 'Cancer', color: 'from-red-500 to-pink-600' },
    { id: 'diabetes', name: 'Diabetes', color: 'from-amber-500 to-orange-600' },
    { id: 'parkinson', name: "Parkinson's", color: 'from-blue-500 to-cyan-600' },
    { id: 'cardiovascular', name: 'Cardio', color: 'from-rose-500 to-red-600' },
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
        <div className="bg-white p-6 rounded-xl border border-slate-200">
            <h3 className="text-lg font-bold text-slate-900 mb-4">🧬 AI Prediction</h3>

            {/* Disease Selector */}
            <div className="flex flex-wrap gap-2 mb-4">
                {DISEASES.map(d => (
                    <button
                        key={d.id}
                        onClick={() => setDisease(d.id)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${disease === d.id
                                ? `bg-gradient-to-r ${d.color} text-white`
                                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                            }`}
                    >
                        {d.name}
                    </button>
                ))}
            </div>

            {/* SMILES Input */}
            <input
                type="text"
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="Enter SMILES..."
                className="w-full px-4 py-2 border border-slate-200 rounded-lg text-sm mb-3 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />

            <button
                onClick={handlePredict}
                disabled={loading || !smiles.trim()}
                className={`w-full py-2 rounded-lg font-semibold text-sm transition-all ${loading
                        ? 'bg-slate-200 text-slate-400'
                        : `bg-gradient-to-r ${selectedDisease?.color} text-white hover:shadow-md`
                    }`}
            >
                {loading ? 'Predicting...' : 'Predict Activity'}
            </button>

            {/* Error */}
            {error && (
                <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-xs">
                    {error}
                </div>
            )}

            {/* Results */}
            {results && results.length > 0 && (
                <div className="mt-4 space-y-3">
                    {results.map((r, idx) => (
                        <div
                            key={idx}
                            className={`p-4 rounded-lg ${r.prediction === 'Active' ? 'bg-green-50 border border-green-200' :
                                    r.prediction === 'Moderate' ? 'bg-yellow-50 border border-yellow-200' :
                                        'bg-red-50 border border-red-200'
                                }`}
                        >
                            <div className="flex justify-between items-center">
                                <span className={`px-2 py-0.5 rounded text-xs font-bold ${r.prediction === 'Active' ? 'bg-green-500 text-white' :
                                        r.prediction === 'Moderate' ? 'bg-yellow-500 text-black' :
                                            'bg-red-500 text-white'
                                    }`}>
                                    {r.prediction}
                                </span>
                                <span className="text-lg font-bold text-slate-900">
                                    {(r.score * 100).toFixed(1)}%
                                </span>
                            </div>
                            {r.interpretation && (
                                <p className="mt-2 text-xs text-slate-600">{r.interpretation}</p>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
