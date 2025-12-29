import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function PlanAPage() {
    const navigate = useNavigate();
    const [disease, setDisease] = useState('');
    const [target, setTarget] = useState('');
    const [smiles, setSmiles] = useState('');

    const handleRunScreening = () => {
        // Logic to start screening job
        console.log("Running Screening for:", { disease, target, smiles });
        navigate('/dashboard'); // Go to dashboard to see results
    };

    return (
        <div className="max-w-7xl mx-auto px-4 py-12">
            <div className="mb-8">
                <button onClick={() => navigate('/')} className="text-slate-500 hover:text-primary-600 mb-4 flex items-center gap-1 text-sm font-medium">
                    ← Back to Plans
                </button>
                <div className="flex items-center gap-3 mb-2">
                    <span className="bg-primary-600 text-white px-3 py-1 rounded-md text-sm font-bold">PLAN A</span>
                    <h1 className="text-3xl font-bold text-slate-900">Disease-First CNS Ensemble</h1>
                </div>
                <p className="text-slate-600 text-lg max-w-3xl">
                    Deploy a federated ensemble of **ChemBERTa (Semantic)**, **GNN (Structure-Aware)**, and **DeepDTA (Binding Affinity)** models to screen millions of compounds against Alzheimer's targets.
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 space-y-8">
                    {/* Step 1: Configuration */}
                    <div className="bg-white border boundary-slate-200 rounded-xl p-8 shadow-sm">
                        <h2 className="text-xl font-semibold text-slate-900 mb-6">1. Configuration</h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-2">Target Disease</label>
                                <select
                                    className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all"
                                    value={disease}
                                    onChange={(e) => setDisease(e.target.value)}
                                >
                                    <option value="">Select Disease...</option>
                                    <option value="alzheimers">Alzheimer's Disease</option>
                                    <option value="parkinsons">Parkinson's Disease</option>
                                    <option value="als">ALS</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-2">Primary Protein Target</label>
                                <select
                                    className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all"
                                    value={target}
                                    onChange={(e) => setTarget(e.target.value)}
                                >
                                    <option value="">Select Target...</option>
                                    <option value="ache">AChE (Acetylcholinesterase)</option>
                                    <option value="bace1">BACE1 (Beta-secretase 1)</option>
                                    <option value="gsk3b">GSK-3β</option>
                                    <option value="ensemble">Ensemble (All 3 Targets)</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Step 2: Compound Input */}
                    <div className="bg-white border boundary-slate-200 rounded-xl p-8 shadow-sm">
                        <h2 className="text-xl font-semibold text-slate-900 mb-6">2. Compound Input</h2>
                        <div className="mb-4">
                            <label className="block text-sm font-medium text-slate-700 mb-2">SMILES Input</label>
                            <textarea
                                className="w-full p-4 border border-slate-300 rounded-lg min-h-[150px] font-mono text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all"
                                placeholder="Paste SMILES strings here (one per line)...&#10;Example:&#10;CCO...&#10;CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
                                value={smiles}
                                onChange={(e) => setSmiles(e.target.value)}
                            ></textarea>
                        </div>
                        <div className="flex justify-between items-center text-sm text-slate-500">
                            <span>Supported formats: SMILES, InChI</span>
                            <button className="text-primary-600 font-medium hover:underline">Load Example Data</button>
                        </div>
                    </div>

                    {/* Action */}
                    <button
                        onClick={handleRunScreening}
                        disabled={!disease || !target}
                        className="w-full bg-primary-600 text-white py-4 rounded-xl text-lg font-bold hover:bg-primary-700 transition-colors shadow-lg shadow-primary-600/20 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        🚀 Launch AI Screening (Tier 1 Ensembles)
                    </button>
                </div>

                {/* Sidebar Info */}
                <div className="space-y-6">
                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-6">
                        <h3 className="font-semibold text-slate-900 mb-3">Model Architecture</h3>
                        <ul className="space-y-3 text-sm text-slate-600">
                            <li className="flex items-start gap-2">
                                <span className="bg-blue-100 text-blue-700 p-1 rounded">🧠</span>
                                <div>
                                    <strong className="block text-slate-800">ChemBERTa-77M</strong>
                                    Semantic understanding of chemical language.
                                </div>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="bg-purple-100 text-purple-700 p-1 rounded">🕸️</span>
                                <div>
                                    <strong className="block text-slate-800">GNN (GATv2)</strong>
                                    Graph attention networks for atomic connectivity.
                                </div>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="bg-green-100 text-green-700 p-1 rounded">🔌</span>
                                <div>
                                    <strong className="block text-slate-800">DeepDTA Fusion</strong>
                                    Combined embeddings for binding affinity (pK<sub>d</sub>).
                                </div>
                            </li>
                        </ul>
                    </div>

                    <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 text-sm text-yellow-800">
                        <strong>💡 Research Tip:</strong> For highest accuracy, select "Ensemble" target to average predictions across multiple docking sites.
                    </div>
                </div>
            </div>
        </div>
    );
}
