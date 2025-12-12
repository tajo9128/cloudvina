import React, { useState } from 'react';
import { aiService } from '../../services/aiService';

const ToxicityPanel = () => {
    const [smiles, setSmiles] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleCheck = async () => {
        if (!smiles) return;
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            // Backend expects a list of smiles
            const response = await aiService.predictToxicity([smiles]);
            // Backend returns { results: [...] }
            if (response.results && response.results.length > 0) {
                setResult(response.results[0]);
            }
        } catch (err) {
            console.error(err);
            setError("Failed to check toxicity. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    const getRiskColor = (risk) => {
        switch (risk) {
            case 'High': return 'bg-red-100 text-red-700 border-red-200';
            case 'Moderate': return 'bg-yellow-100 text-yellow-700 border-yellow-200';
            case 'Low': return 'bg-green-100 text-green-700 border-green-200';
            default: return 'bg-slate-100 text-slate-700 border-slate-200';
        }
    };

    return (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h2 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                <span className="text-2xl">☣️</span> Toxicity Safety Check
            </h2>

            <div className="flex gap-2 mb-4">
                <input
                    type="text"
                    value={smiles}
                    onChange={(e) => setSmiles(e.target.value)}
                    placeholder="Enter SMILES to check safety (e.g. CCO)"
                    className="flex-1 p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none font-mono text-sm"
                />
                <button
                    onClick={handleCheck}
                    disabled={loading || !smiles}
                    className="bg-slate-800 text-white px-6 py-2 rounded-lg font-medium hover:bg-slate-900 transition-colors disabled:opacity-50"
                >
                    {loading ? 'Scanning...' : 'Scan'}
                </button>
            </div>

            {error && (
                <div className="text-red-600 text-sm mb-4">{error}</div>
            )}

            {result && (
                <div className="animate-fadeIn">
                    <div className={`p-4 rounded-lg border mb-4 flex justify-between items-center ${getRiskColor(result.risk)}`}>
                        <div>
                            <span className="font-bold text-lg">Risk Level: {result.risk}</span>
                            <p className="text-sm opacity-90 mt-1">
                                {result.valid ? (
                                    result.alert_count > 0 ? `${result.alert_count} structural alerts found.` : "No structural alerts found."
                                ) : (
                                    "Invalid Molecule Structure"
                                )}
                            </p>
                        </div>
                        <div className="text-4xl">
                            {result.risk === 'High' ? '🛑' : result.risk === 'Moderate' ? '⚠️' : '✅'}
                        </div>
                    </div>

                    {result.alerts && result.alerts.length > 0 && (
                        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
                            <h4 className="font-medium text-slate-900 mb-2">Detected Alerts:</h4>
                            <ul className="space-y-2">
                                {result.alerts.map((alert, idx) => (
                                    <li key={idx} className="flex items-start gap-2 text-sm text-slate-700">
                                        <span className="font-semibold text-red-600 min-w-[80px]">{alert.type}:</span>
                                        <span>{alert.description}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default ToxicityPanel;
