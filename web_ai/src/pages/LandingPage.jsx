import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Dna, Activity, Zap, Server, Database, Globe } from 'lucide-react';

export default function LandingPage() {
    const navigate = useNavigate();

    return (
        <div className="bg-white min-h-screen font-sans text-slate-900">
            {/* --------------------------------------------------------------------------------
               SECTION 1: HERO (Modern, Image-driven, Strong CTA)
               -------------------------------------------------------------------------------- */}
            <section className="relative overflow-hidden bg-[#f8f9fa] pt-16 pb-24 lg:pt-32 lg:pb-40">
                <div className="absolute top-0 right-0 -translate-y-1/4 translate-x-1/4 w-[800px] h-[800px] bg-gradient-to-br from-primary-100/40 to-teal-100/40 rounded-full blur-3xl opacity-50 pointer-events-none"></div>
                <div className="absolute bottom-0 left-0 translate-y-1/4 -translate-x-1/4 w-[600px] h-[600px] bg-gradient-to-tr from-blue-100/40 to-purple-100/40 rounded-full blur-3xl opacity-50 pointer-events-none"></div>

                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                    <div className="grid lg:grid-cols-2 gap-12 items-center">
                        <div>
                            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-50 border border-teal-100 text-teal-700 text-xs font-bold uppercase tracking-wider mb-6">
                                <span className="w-2 h-2 rounded-full bg-teal-500 animate-pulse"></span>
                                AI-Powered Drug Discovery v2.0
                            </div>
                            <h1 className="text-5xl lg:text-6xl font-bold tracking-tight text-slate-900 mb-6 leading-[1.1]">
                                Accelerate Your <br />
                                <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#218081] to-blue-600">
                                    PhD Research
                                </span>
                            </h1>
                            <p className="text-lg text-slate-600 mb-8 leading-relaxed max-w-lg">
                                Deploy state-of-the-art <strong>ChemBERTa</strong>, <strong>GNN</strong>, and <strong>DeepDTA</strong> models to identify novel therapeutics for Alzheimer's and CNS disorders. From hit identification to formulation optimization.
                            </p>

                            <div className="flex flex-col sm:flex-row gap-4">
                                <button
                                    onClick={() => document.getElementById('plans-section').scrollIntoView({ behavior: 'smooth' })}
                                    className="bg-[#218081] text-white px-8 py-4 rounded-xl font-bold text-lg shadow-lg shadow-teal-700/20 hover:bg-[#1a6468] hover:shadow-xl hover:-translate-y-0.5 transition-all flex items-center justify-center gap-2"
                                >
                                    Start AI Screening <ArrowRight className="w-5 h-5" />
                                </button>
                                <a
                                    href="https://biodockify.com"
                                    className="px-8 py-4 rounded-xl font-bold text-lg border border-slate-200 text-slate-600 hover:bg-white hover:text-slate-900 hover:border-slate-300 transition-all flex items-center justify-center gap-2 bg-white/50 backdrop-blur-sm"
                                >
                                    <Globe className="w-5 h-5" /> Main Website
                                </a>
                            </div>

                            {/* SEO / Trust Badges */}
                            <div className="mt-10 flex flex-wrap gap-6 text-sm text-slate-500 font-medium">
                                <div className="flex items-center gap-2">
                                    <Server className="w-4 h-4 text-slate-400" /> AutoDock Vina
                                </div>
                                <div className="flex items-center gap-2">
                                    <Database className="w-4 h-4 text-slate-400" /> ChEMBL Integrated
                                </div>
                                <div className="flex items-center gap-2">
                                    <Zap className="w-4 h-4 text-slate-400" /> GPU Acceleration
                                </div>
                            </div>
                        </div>

                        {/* Hero Image */}
                        <div className="relative hidden lg:block">
                            <div className="absolute inset-0 bg-gradient-to-tr from-[#218081]/10 to-blue-500/10 rounded-2xl transform rotate-3"></div>
                            <img
                                src="/images/hero_ai.png"
                                alt="AI Molecular Discovery Visualization"
                                className="relative rounded-2xl shadow-2xl border border-white/20 transform -rotate-2 hover:rotate-0 transition-transform duration-700 w-full object-cover"
                            />
                            {/* Floating Card */}
                            <div className="absolute -bottom-8 -left-8 bg-white p-4 rounded-xl shadow-xl border border-slate-100 flex items-center gap-4 animate-count-up">
                                <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center text-2xl">🌿</div>
                                <div>
                                    <div className="text-xs text-slate-500 font-bold uppercase">Active Compounds</div>
                                    <div className="text-xl font-bold text-slate-900">1,248 Found</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* --------------------------------------------------------------------------------
               SECTION 2: RESEARCH PLANS (Plan A vs Plan B)
               -------------------------------------------------------------------------------- */}
            <section id="plans-section" className="py-24 bg-white relative">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-16 max-w-2xl mx-auto">
                        <h2 className="text-3xl font-bold text-slate-900 mb-4">Choose Your Research Pathway</h2>
                        <p className="text-slate-600">
                            Select the workflow that matches your PhD objectives. Whether you are starting with a disease target or investigating a medicinal plant.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-8 lg:gap-12">
                        {/* PLAN A CARD */}
                        <div className="group relative bg-white border border-slate-200 rounded-2xl p-8 hover:border-[#218081] transition-all hover:shadow-2xl hover:shadow-teal-900/5">
                            <div className="absolute top-0 right-0 bg-slate-100 text-slate-600 text-xs font-bold px-3 py-1 rounded-bl-xl rounded-tr-xl">
                                DISEASE-FIRST
                            </div>
                            <div className="w-14 h-14 bg-cyan-100 rounded-xl flex items-center justify-center text-3xl mb-6 group-hover:scale-110 transition-transform">
                                🧠
                            </div>
                            <h3 className="text-2xl font-bold text-slate-900 mb-2 group-hover:text-[#218081] transition-colors">
                                Plan A: CNS Ensemble
                            </h3>
                            <p className="text-slate-500 mb-6 min-h-[48px]">
                                Federated screening of synthetic libraries against Alzheimer's targets (AChE, BACE1) using ChemBERTa & GNNs.
                            </p>

                            <ul className="space-y-3 mb-8">
                                <li className="flex items-start gap-3 text-sm text-slate-600">
                                    <span className="text-cyan-500 font-bold">✓</span> Target-based Virtual Screening
                                </li>
                                <li className="flex items-start gap-3 text-sm text-slate-600">
                                    <span className="text-cyan-500 font-bold">✓</span> Multi-Model Consensus Scoring
                                </li>
                                <li className="flex items-start gap-3 text-sm text-slate-600">
                                    <span className="text-cyan-500 font-bold">✓</span> High-Throughput Docking (HTVS)
                                </li>
                            </ul>

                            <button
                                onClick={() => navigate('/project/cns')}
                                className="w-full py-3 border-2 border-[#218081] text-[#218081] font-bold rounded-lg hover:bg-[#218081] hover:text-white transition-all"
                            >
                                Select Plan A
                            </button>
                        </div>

                        {/* PLAN B CARD */}
                        <div className="group relative bg-white border border-slate-200 rounded-2xl p-8 hover:border-[#209a66] transition-all hover:shadow-2xl hover:shadow-green-900/5">
                            <div className="absolute top-0 right-0 bg-slate-100 text-slate-600 text-xs font-bold px-3 py-1 rounded-bl-xl rounded-tr-xl">
                                PLANT-FIRST
                            </div>
                            <div className="w-14 h-14 bg-green-100 rounded-xl flex items-center justify-center text-3xl mb-6 group-hover:scale-110 transition-transform">
                                🌿
                            </div>
                            <h3 className="text-2xl font-bold text-slate-900 mb-2 group-hover:text-[#209a66] transition-colors">
                                Plan B: Phytochemicals
                            </h3>
                            <p className="text-slate-500 mb-6 min-h-[48px]">
                                <strong>Phases 1-3</strong>: Upload GC-MS data to identify bioactives and predict their SAR activity instantly.
                            </p>

                            <ul className="space-y-3 mb-8">
                                <li className="flex items-start gap-3 text-sm text-slate-600">
                                    <span className="text-green-500 font-bold">✓</span> GC-MS Peak Annotation
                                </li>
                                <li className="flex items-start gap-3 text-sm text-slate-600">
                                    <span className="text-green-500 font-bold">✓</span> AI Phytochemical ID
                                </li>
                                <li className="flex items-start gap-3 text-sm text-slate-600">
                                    <span className="text-green-500 font-bold">✓</span> Thesis Chapter Generation
                                </li>
                            </ul>

                            <button
                                onClick={() => navigate('/project/phytochemicals')}
                                className="w-full py-3 border-2 border-[#209a66] text-[#209a66] font-bold rounded-lg hover:bg-[#209a66] hover:text-white transition-all"
                            >
                                Select Plan B
                            </button>
                        </div>
                    </div>
                </div>
            </section>

            {/* --------------------------------------------------------------------------------
               SECTION 3: FORMULATION AI (Feature Highlight)
               -------------------------------------------------------------------------------- */}
            <section className="py-24 bg-slate-900 text-white relative overflow-hidden">
                <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10"></div>
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                    <div className="grid lg:grid-cols-2 gap-16 items-center">
                        <div className="order-2 lg:order-1">
                            <div className="inline-block px-3 py-1 rounded bg-purple-500/20 text-purple-300 text-xs font-bold mb-4 border border-purple-500/30">
                                PILLAR 6: AI FORMULATION
                            </div>
                            <h2 className="text-3xl md:text-4xl font-bold mb-6">
                                BioDockify-Formulate™ <br />
                                <span className="text-purple-400">7-Model Stack</span>
                            </h2>
                            <p className="text-slate-300 text-lg mb-8 leading-relaxed">
                                Don't stop at lead identification. Use our federated AI stack to optimize your drug's delivery system. Predict <strong>Pre-formulation Risks</strong>, select the best <strong>Excipients</strong>, and estimate <strong>Shelf-life Stability</strong> using DeepSurv.
                            </p>

                            <div className="grid grid-cols-2 gap-6 mb-8">
                                <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                                    <h4 className="font-bold text-white mb-1">Model 5: Stability</h4>
                                    <p className="text-xs text-slate-400">DeepSurv (Cox Proportional Hazards)</p>
                                </div>
                                <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                                    <h4 className="font-bold text-white mb-1">Model 1B: Solid State</h4>
                                    <p className="text-xs text-slate-400">GNN for Polymorphism Risk</p>
                                </div>
                            </div>

                            <button
                                onClick={() => navigate('/formulation')}
                                className="bg-purple-600 hover:bg-purple-700 text-white px-8 py-3 rounded-lg font-bold transition-all flex items-center gap-2"
                            >
                                <Activity className="w-5 h-5" /> Open Formulation Studio
                            </button>
                        </div>

                        <div className="order-1 lg:order-2">
                            {/* Abstract Visualization of the 7-Model Stack */}
                            <div className="relative">
                                <div className="absolute inset-0 bg-purple-500 blur-[100px] opacity-20"></div>
                                <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-2xl p-8 transform rotate-2">
                                    <div className="space-y-4">
                                        {[
                                            "1. API Representation (ChemBERTa)",
                                            "2. Pre-Formulation Risk (XGBoost)",
                                            "3. Excipient Selection (Ranker)",
                                            "4. Dissolution Profile (LSTM)",
                                            "5. Stability Survival (DeepSurv)",
                                            "6. QbD Documentation (LLM)",
                                            "7. ANDA Readiness (Ensemble)"
                                        ].map((item, i) => (
                                            <div key={i} className="flex items-center gap-4 bg-slate-900/50 p-3 rounded border border-slate-700/50">
                                                <div className={`w-2 h-2 rounded-full ${i === 6 ? 'bg-green-500' : 'bg-purple-500'}`}></div>
                                                <span className="font-mono text-sm text-slate-300">{item}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* --------------------------------------------------------------------------------
               SEO / FOOTER CTA
               -------------------------------------------------------------------------------- */}
            <section className="bg-slate-50 border-t border-slate-200 py-16">
                <div className="max-w-4xl mx-auto px-4 text-center">
                    <h3 className="text-xl font-semibold text-slate-900 mb-4">
                        Part of the BioDockify Research Ecosystem
                    </h3>
                    <p className="text-slate-600 mb-8">
                        BioDockify AI Suite is optimized for academic research, providing molecular docking, QSAR modeling, and formulation AI in one unified platform.
                    </p>
                    <a
                        href="https://biodockify.com"
                        className="inline-flex items-center gap-2 text-[#218081] font-bold hover:underline"
                    >
                        Visit Main Website for Standard Docking <ArrowRight className="w-4 h-4" />
                    </a>
                </div>
            </section>
        </div>
    );
}
