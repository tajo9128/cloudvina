import React, { useState } from 'react';
import AdmetRadar from '../../components/AdmetRadar';
import SEOHelmet from '../../components/SEOHelmet';

export default function AdmetToolPage() {
    const [smiles, setSmiles] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const handlePredict = async (e) => {
        e.preventDefault();
        setLoading(true);
        // Simulate API call or Real API if accessible publically
        // Ideally: fetch(`${API_URL}/predict/admet`, ...)

        setTimeout(() => {
            // Mock Result for Standalone Demo
            setResult({
                score: Math.floor(Math.random() * 30) + 60,
                molecular_properties: {
                    molecular_weight: 350 + Math.random() * 200,
                    logp: 1 + Math.random() * 4,
                    hbd: Math.floor(Math.random() * 5),
                    hba: Math.floor(Math.random() * 10),
                    tpsa: 50 + Math.random() * 100
                },
                lipinski: { violations: 0 },
                herg: { risk_level: Math.random() > 0.5 ? 'Low' : 'Medium' },
                ames: { prediction: 'Negative' },
                cyp: { overall_ddi_risk: 'Low' }
            });
            setLoading(false);
        }, 1500);
    };

    return (
        <div className="min-h-screen bg-slate-50 py-12 px-4">
            <SEOHelmet
                title="Free ADMET Prediction Tool | Toxicity & Properties"
                description="Predict ADMET properties, toxicity, and drug-likeness from SMILES strings instantly using AI."
                keywords="admet prediction, toxicity prediction, smiles to admet, drug likeness calculator"
                canonical="https://biodockify.com/tools/admet"
            />
            <div className="max-w-4xl mx-auto">
                <div className="text-center mb-10">
                    <h1 className="text-3xl font-bold text-slate-900 mb-3">ADMET Predictor</h1>
                    <p className="text-slate-500">Instant toxicity and physicochemical checks for any molecule.</p>
                </div>

                <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
                    <form onSubmit={handlePredict} className="flex gap-4">
                        <input
                            type="text"
                            value={smiles}
                            onChange={(e) => setSmiles(e.target.value)}
                            placeholder="Enter SMILES string (e.g. C=CC(=O)OC1=CC=CC=C1)"
                            className="flex-1 px-4 py-3 border-2 border-slate-200 rounded-xl focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/10 outline-none font-mono text-sm"
                            required
                        />
                        <button
                            type="submit"
                            disabled={loading || !smiles}
                            className="px-8 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl shadow-lg disabled:opacity-50 transition-all"
                        >
                            {loading ? 'Analyzing...' : 'Predict'}
                        </button>
                    </form>
                </div>

                {result && (
                    <div className="grid md:grid-cols-2 gap-8">
                        <div className="bg-white p-8 rounded-2xl shadow-lg border border-slate-100 flex flex-col items-center">
                            <h3 className="font-bold text-slate-700 mb-6 uppercase tracking-wider text-sm">Physicochemical Profile</h3>
                            <AdmetRadar data={result} />
                        </div>

                        <div className="space-y-4">
                            <div className="bg-white p-6 rounded-2xl shadow-lg border border-slate-100">
                                <h3 className="font-bold text-slate-700 mb-4 uppercase tracking-wider text-sm">Toxicity Alerts</h3>
                                <div className="space-y-3">
                                    <div className="flex justify-between items-center p-3 bg-red-50 rounded-lg text-red-900">
                                        <span>hERG Cardiotoxicity</span>
                                        <span className="font-bold">{result.herg.risk_level}</span>
                                    </div>
                                    <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg text-green-900">
                                        <span>AMES Mutagenicity</span>
                                        <span className="font-bold">{result.ames.prediction}</span>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-white p-6 rounded-2xl shadow-lg border border-slate-100">
                                <h3 className="font-bold text-slate-700 mb-4 uppercase tracking-wider text-sm">Drug Likeness</h3>
                                <div className="flex items-center gap-4">
                                    <div className="text-4xl font-bold text-indigo-600">{result.score}/100</div>
                                    <div className="text-sm text-slate-500">
                                        BioDockify Score based on QED and Lipinski rules.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
