import React from 'react';
import { Link } from 'react-router-dom';
import { Activity, Brain, ShieldAlert, Zap, ArrowRight, Dna, Database, Layers } from 'lucide-react';

export default function LandingPage() {
    return (
        <div className="min-h-screen bg-slate-50 font-sans">
            {/* Hero Section */}
            <div className="relative bg-slate-900 overflow-hidden">
                <div className="absolute inset-0">
                    <img
                        src="/assets/ai_hero.png"
                        alt="AI Drug Discovery"
                        className="w-full h-full object-cover opacity-20"
                    />
                    <div className="absolute inset-0 bg-gradient-to-b from-slate-900/10 via-slate-900/50 to-slate-900"></div>
                </div>

                <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-32 pb-24 text-center">
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 rounded-full text-sm font-medium mb-8 backdrop-blur-sm">
                        <Zap className="w-4 h-4" />
                        <span className="font-semibold text-indigo-200">New Engine:</span> Toxicity Prediction Live
                    </div>

                    <h1 className="text-5xl md:text-7xl font-bold font-display text-white mb-8 tracking-tight">
                        Generative AI for <br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">
                            Drug Discovery
                        </span>
                    </h1>

                    <p className="text-xl text-slate-300 max-w-3xl mx-auto mb-12 leading-relaxed">
                        Accelerate your research with our <strong>Auto-QSAR</strong> and <strong>Toxicity Prediction (Phase 4)</strong> engines.
                        From molecular docking to safety profiling in one platform.
                    </p>

                    <div className="flex flex-wrap justify-center gap-4">
                        <Link
                            to="/dashboard"
                            className="px-8 py-4 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-xl transition-all shadow-lg shadow-indigo-900/20 flex items-center gap-2"
                        >
                            <Brain className="w-5 h-5" />
                            Launch Workspace
                        </Link>
                        <a
                            href="#features"
                            className="px-8 py-4 bg-white/10 hover:bg-white/20 text-white font-semibold rounded-xl transition-all backdrop-blur-sm border border-white/10"
                        >
                            Explore Features
                        </a>
                    </div>
                </div>
            </div>

            {/* AI Capabilities Section */}
            <div id="features" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
                <div className="text-center mb-16">
                    <h2 className="text-3xl font-bold font-display text-slate-900 mb-4">
                        The AI BioDockify Advantage
                    </h2>
                    <p className="text-slate-600 max-w-2xl mx-auto">
                        We've integrated state-of-the-art machine learning models directly into your docking workflow.
                    </p>
                </div>

                <div className="grid md:grid-cols-3 gap-8">
                    {/* Feature 1: QSAR (Recent AI Plan) */}
                    <div className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm hover:shadow-lg transition-all group relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-3 bg-gradient-to-bl from-indigo-50 to-transparent rounded-bl-3xl">
                            <Dna className="w-5 h-5 text-indigo-600" />
                        </div>
                        <div className="w-14 h-14 bg-indigo-100 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                            <Brain className="w-7 h-7 text-indigo-600" />
                        </div>
                        <h3 className="text-xl font-bold text-slate-900 mb-3">Auto-QSAR Engine</h3>
                        <p className="text-slate-600 leading-relaxed">
                            <strong>Recent Update:</strong> Train custom regression and classification models using ChemBERTa. Predict bioactivity (IC50) with high accuracy.
                        </p>
                    </div>

                    {/* Feature 2: Toxicity (Phase 4) */}
                    <div className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm hover:shadow-lg transition-all group">
                        <div className="w-14 h-14 bg-emerald-100 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                            <ShieldAlert className="w-7 h-7 text-emerald-600" />
                        </div>
                        <h3 className="text-xl font-bold text-slate-900 mb-3">Toxicity Prediction</h3>
                        <p className="text-slate-600 leading-relaxed">
                            <strong>Phase 4 Integration:</strong> Early-stage ADMET profiling. Instantly screen compounds for mutagenicity, hepatotoxicity, and hERG inhibition.
                        </p>
                    </div>

                    {/* Feature 3: Data Management */}
                    <div className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm hover:shadow-lg transition-all group">
                        <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                            <Database className="w-7 h-7 text-blue-600" />
                        </div>
                        <div className="flex items-center gap-2 mb-3">
                            <h3 className="text-xl font-bold text-slate-900">Project Hub</h3>
                            <span className="px-2 py-0.5 bg-slate-100 text-slate-600 text-xs font-bold rounded">CORE</span>
                        </div>
                        <p className="text-slate-600 leading-relaxed">
                            Centralized data management for all your target proteins, ligands, and experiment results.
                        </p>
                    </div>
                </div>
            </div>

            {/* How It Works */}
            <div className="bg-white py-24 border-y border-slate-200">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex flex-col md:flex-row items-center gap-16">
                        <div className="md:w-1/2">
                            <h2 className="text-3xl font-bold font-display text-slate-900 mb-6">
                                From 2D Structure to <br />
                                3D Insight
                            </h2>
                            <div className="space-y-8">
                                <div className="flex gap-4">
                                    <div className="w-8 h-8 rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold flex-shrink-0">1</div>
                                    <div>
                                        <h4 className="font-bold text-slate-900">Define Project</h4>
                                        <p className="text-slate-600 text-sm mt-1">Create a new workspace for your target protein (e.g., EGFR, COX-2).</p>
                                    </div>
                                </div>
                                <div className="flex gap-4">
                                    <div className="w-8 h-8 rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold flex-shrink-0">2</div>
                                    <div>
                                        <h4 className="font-bold text-slate-900">Upload & Dock</h4>
                                        <p className="text-slate-600 text-sm mt-1">Submit ligands for AutoDock Vina processing via our robust backend.</p>
                                    </div>
                                </div>
                                <div className="flex gap-4">
                                    <div className="w-8 h-8 rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold flex-shrink-0">3</div>
                                    <div>
                                        <h4 className="font-bold text-slate-900">Analyze & Predict</h4>
                                        <p className="text-slate-600 text-sm mt-1">Run QSAR models to predict activity and check toxicity flags instantly.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="md:w-1/2">
                            <div className="bg-slate-900 rounded-2xl p-6 shadow-2xl shadow-indigo-500/10 border border-slate-800">
                                <div className="flex items-center gap-2 mb-4 border-b border-slate-800 pb-4">
                                    <div className="w-3 h-3 rounded-full bg-red-500"></div>
                                    <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                                    <span className="text-slate-500 text-xs ml-2">Toxicity Analysis Report</span>
                                </div>
                                <div className="space-y-4 font-mono text-sm">
                                    <div className="flex justify-between items-center text-slate-300">
                                        <span>Mutagenicity</span>
                                        <span className="text-green-400">Negative (98%)</span>
                                    </div>
                                    <div className="w-full bg-slate-800 h-1 rounded-full"><div className="w-[2%] bg-green-500 h-1 rounded-full"></div></div>

                                    <div className="flex justify-between items-center text-slate-300">
                                        <span>hERG Inhibition</span>
                                        <span className="text-yellow-400">Low Risk (24%)</span>
                                    </div>
                                    <div className="w-full bg-slate-800 h-1 rounded-full"><div className="w-[24%] bg-yellow-500 h-1 rounded-full"></div></div>

                                    <div className="flex justify-between items-center text-slate-300">
                                        <span>Hepatotoxicity</span>
                                        <span className="text-green-400">Negative (12%)</span>
                                    </div>
                                    <div className="w-full bg-slate-800 h-1 rounded-full"><div className="w-[12%] bg-green-500 h-1 rounded-full"></div></div>
                                </div>
                                <div className="mt-6 pt-4 border-t border-slate-800 text-center">
                                    <span className="text-indigo-400 text-xs">AI Inference time: 0.12s</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* CTA */}
            <div className="bg-indigo-900 text-white py-20">
                <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                    <h2 className="text-3xl font-bold font-display mb-6">
                        Ready to Transform Your Research?
                    </h2>
                    <Link
                        to="/dashboard"
                        className="inline-block px-8 py-4 bg-white text-indigo-900 font-bold rounded-lg hover:bg-indigo-50 transition-colors shadow-2xl"
                    >
                        Go to Dashboard
                    </Link>
                </div>
            </div>
        </div>
    );
}
