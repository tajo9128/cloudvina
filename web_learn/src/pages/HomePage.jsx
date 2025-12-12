import React from 'react';
import { Link } from 'react-router-dom';
import { GraduationCap, BookOpen, Users, Zap, Activity, Brain, ShieldAlert } from 'lucide-react';

export default function HomePage() {
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
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary-500/10 border border-primary-500/20 text-primary-300 rounded-full text-sm font-medium mb-8 backdrop-blur-sm">
                        <Zap className="w-4 h-4" />
                        <span className="font-semibold text-primary-200">New Feature:</span> AI Toxicity Prediction
                    </div>

                    <h1 className="text-5xl md:text-7xl font-bold font-display text-white mb-8 tracking-tight">
                        The Future of <br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-indigo-400">
                            AI Drug Discovery
                        </span>
                    </h1>

                    <p className="text-xl text-slate-300 max-w-3xl mx-auto mb-12 leading-relaxed">
                        BioDockify combines advanced molecular docking with <strong className="text-white">Auto-QSAR</strong> and <strong className="text-white">Toxicity Prediction</strong> (Phase 4). Learn, build, and deploy your own drug discovery pipelines.
                    </p>

                    <div className="flex flex-wrap justify-center gap-4">
                        <a
                            href="https://ai.biodockify.com"
                            className="px-8 py-4 bg-primary-600 hover:bg-primary-500 text-white font-semibold rounded-xl transition-all shadow-lg shadow-primary-900/20 flex items-center gap-2"
                        >
                            <Brain className="w-5 h-5" />
                            Launch AI Platform
                        </a>
                        <Link
                            to="/courses"
                            className="px-8 py-4 bg-white/10 hover:bg-white/20 text-white font-semibold rounded-xl transition-all backdrop-blur-sm border border-white/10"
                        >
                            Start Learning
                        </Link>
                    </div>
                </div>
            </div>

            {/* AI Capabilities Section (Phase 4 & Recent Plan) */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
                <div className="text-center mb-16">
                    <h2 className="text-3xl font-bold font-display text-slate-900 mb-4">
                        Comprehensive Design Suite
                    </h2>
                    <p className="text-slate-600 max-w-2xl mx-auto">
                        From initial docking to safety profiling, our platform integrates the latest AI tools into your workflow.
                    </p>
                </div>

                <div className="grid md:grid-cols-3 gap-8">
                    {/* Feature 1: Docking */}
                    <div className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm hover:shadow-lg transition-all group">
                        <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                            <Activity className="w-7 h-7 text-blue-600" />
                        </div>
                        <h3 className="text-xl font-bold text-slate-900 mb-3">Molecular Docking</h3>
                        <p className="text-slate-600 leading-relaxed">
                            Perform high-throughput virtual screening using AutoDock Vina in the cloud. Visualize protein-ligand interactions in 3D.
                        </p>
                    </div>

                    {/* Feature 2: QSAR (AI Plan) */}
                    <div className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm hover:shadow-lg transition-all group relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-3 bg-gradient-to-bl from-primary-100 to-transparent rounded-bl-3xl">
                             <Zap className="w-5 h-5 text-primary-600" />
                        </div>
                        <div className="w-14 h-14 bg-indigo-100 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                            <Brain className="w-7 h-7 text-indigo-600" />
                        </div>
                        <h3 className="text-xl font-bold text-slate-900 mb-3">Auto-QSAR Engine</h3>
                        <p className="text-slate-600 leading-relaxed">
                            Train custom QSAR models using ChemBERTa and Random Forest. Predict bioactivity with state-of-the-art accuracy.
                        </p>
                    </div>

                    {/* Feature 3: Toxicity (Phase 4) */}
                    <div className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm hover:shadow-lg transition-all group">
                        <div className="w-14 h-14 bg-emerald-100 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                            <ShieldAlert className="w-7 h-7 text-emerald-600" />
                        </div>
                        <h3 className="text-xl font-bold text-slate-900 mb-3">Toxicity Prediction</h3>
                        <p className="text-slate-600 leading-relaxed">
                            <strong>Phase 4 Integration:</strong> Early-stage ADMET profiling. Screen compounds for potential toxicity before synthesis.
                        </p>
                    </div>
                </div>
            </div>

            {/* Learning Resources (Existing Content Refined) */}
            <div className="bg-slate-100 py-24">
                 <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex flex-col md:flex-row items-center justify-between gap-12">
                        <div className="md:w-1/2">
                            <h2 className="text-3xl font-bold font-display text-slate-900 mb-6">
                                Master the Tools
                            </h2>
                            <p className="text-lg text-slate-600 mb-8 leading-relaxed">
                                Don't just run the software—understand the science. Our comprehensive courses cover everything from PDB file preparation to analyzing binding free energies.
                            </p>
                            <ul className="space-y-4 mb-8">
                                <li className="flex items-center gap-3 text-slate-700">
                                    <div className="bg-green-100 p-1 rounded-full"><div className="w-2 h-2 bg-green-500 rounded-full"></div></div>
                                    Interactive Video Tutorials
                                </li>
                                <li className="flex items-center gap-3 text-slate-700">
                                    <div className="bg-green-100 p-1 rounded-full"><div className="w-2 h-2 bg-green-500 rounded-full"></div></div>
                                    Step-by-step Project Guides
                                </li>
                                <li className="flex items-center gap-3 text-slate-700">
                                    <div className="bg-green-100 p-1 rounded-full"><div className="w-2 h-2 bg-green-500 rounded-full"></div></div>
                                    Certificate of Completion
                                </li>
                            </ul>
                            <Link
                                to="/courses"
                                className="inline-flex items-center gap-2 text-primary-600 font-semibold hover:text-primary-700 transition-colors group"
                            >
                                Browne Course Catalog <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                            </Link>
                        </div>
                        <div className="md:w-1/2 relative">
                             <div className="absolute -inset-4 bg-gradient-to-r from-primary-500 to-indigo-500 rounded-2xl opacity-20 blur-2xl"></div>
                             <div className="relative bg-white rounded-2xl shadow-xl p-8 border border-slate-200">
                                 <div className="flex items-center gap-4 mb-6">
                                     <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center">
                                         <BookOpen className="w-6 h-6 text-orange-600" />
                                     </div>
                                     <div>
                                         <h4 className="font-bold text-slate-900">Featured Course</h4>
                                         <p className="text-sm text-slate-500">Beginner Friendly</p>
                                     </div>
                                 </div>
                                 <h3 className="text-xl font-bold mb-2">Introduction to AutoDock Vina</h3>
                                 <p className="text-slate-600 text-sm mb-4">
                                     Learn the basics of molecular docking, from ligand preparation to result interpretation.
                                 </p>
                                 <div className="w-full bg-slate-100 rounded-full h-2 mb-4">
                                     <div className="bg-orange-500 h-2 rounded-full w-2/3"></div>
                                 </div>
                                 <div className="flex justify-between text-xs text-slate-500">
                                     <span>12 Lessons</span>
                                     <span>4.5 Hours</span>
                                 </div>
                             </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* CTA Section */}
            <div className="bg-slate-900 text-white py-24 border-t border-slate-800">
                <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                    <h2 className="text-3xl md:text-4xl font-bold font-display mb-6">
                        Start Your Research Journey
                    </h2>
                    <p className="text-xl text-slate-400 mb-10 max-w-2xl mx-auto">
                        Whether you're a student learning the ropes or a researcher screening thousands of compounds, BioDockify has the tools you need.
                    </p>
                    <div className="flex flex-col sm:flex-row justify-center gap-4">
                         <Link
                            to="/signup"
                            className="px-8 py-4 bg-primary-600 hover:bg-primary-500 text-white font-semibold rounded-lg transition-colors shadow-lg shadow-primary-900/50"
                        >
                            Create Free Account
                        </Link>
                         <Link
                            to="/community"
                            className="px-8 py-4 bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold rounded-lg transition-colors border border-slate-700"
                        >
                            Join Community
                        </Link>
                    </div>
                </div>
            </div>
        </div >
    );
}

function ArrowRight({ className }) {
    return (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
    )
}

