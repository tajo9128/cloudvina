import React from 'react';
import { ArrowRight, Download, Brain, Network, Database, Layers, Search, FileText, CheckCircle, Cpu, Shield, Globe } from 'lucide-react';
import { Link } from 'react-router-dom';

const PharmaResearchAIPage = () => {
    return (
        <div className="min-h-screen bg-white font-sans text-slate-900">
            {/* Hero Section */}
            <div className="relative overflow-hidden bg-slate-900 text-white">
                <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?q=80&w=2070&auto=format&fit=crop')] bg-cover bg-center opacity-20"></div>
                <div className="absolute inset-0 bg-gradient-to-r from-slate-900 via-slate-900/90 to-transparent"></div>

                <div className="relative max-w-7xl mx-auto px-6 py-24 lg:py-32">
                    <div className="max-w-3xl">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/20 border border-indigo-400/30 text-indigo-300 text-sm font-medium mb-6">
                            <Brain size={16} />
                            <span>BioDockify Pharma Research AI</span>
                        </div>
                        <h1 className="text-5xl lg:text-7xl font-bold tracking-tight mb-8">
                            Where AI becomes a <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">research collaborator</span>, not just a tool.
                        </h1>
                        <p className="text-xl text-slate-300 mb-10 leading-relaxed">
                            An Autonomous Agent-Based AI Platform for End-to-End Pharmaceutical Research.
                            Support your journey from ideation to thesis-ready outputs with evidence-constrained reasoning and human-in-the-loop oversight.
                        </p>
                        <div className="flex flex-col sm:flex-row gap-4">
                            <a
                                href="https://github.com/tajo9128/BioDockify-pharma-research-ai/releases"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center justify-center px-8 py-4 text-lg font-bold rounded-full bg-indigo-600 hover:bg-indigo-500 text-white transition-all shadow-lg hover:shadow-indigo-500/30"
                            >
                                <Download className="mr-2" size={20} />
                                Download Latest Release
                            </a>
                            <Link to="/contact" className="inline-flex items-center justify-center px-8 py-4 text-lg font-bold rounded-full bg-white/10 hover:bg-white/20 text-white border border-white/10 transition-all backdrop-blur-sm">
                                Request Demo
                            </Link>
                        </div>
                        <p className="mt-6 text-sm text-slate-400">
                            * Available for Windows, Linux, and Docker • v3.5 Stable
                        </p>
                    </div>
                </div>
            </div>

            {/* Overview Section */}
            <section className="py-20 px-6 max-w-7xl mx-auto">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
                    <div>
                        <h2 className="text-3xl lg:text-4xl font-bold mb-6">Overview</h2>
                        <p className="text-lg text-slate-600 leading-relaxed mb-6">
                            BioDockify Pharma Research AI is an autonomous, agent-based research intelligence platform designed to support pharmaceutical and life-science research from ideation to thesis-ready outputs.
                        </p>
                        <p className="text-lg text-slate-600 leading-relaxed mb-6">
                            Unlike conventional AI tools that assist isolated tasks, BioDockify functions as a persistent research assistant, capable of planning, executing, validating, and documenting complex pharmaceutical research workflows—all with human-in-the-loop scientific oversight.
                        </p>
                        <p className="text-lg text-slate-600 leading-relaxed">
                            The platform is built on <strong>Agent Zero orchestration principles</strong>, Docker-based execution, and evidence-constrained AI reasoning, making it ideal for PhD research, academic labs, and early-stage drug discovery teams.
                        </p>
                    </div>
                    <div className="bg-slate-50 p-8 rounded-3xl border border-slate-100">
                        <h3 className="text-xl font-bold mb-6 text-slate-800">Why BioDockify?</h3>
                        <ul className="space-y-4">
                            {[
                                "Fragmented tools and workflows",
                                "Manual, time-intensive literature reviews",
                                "Weak integration between AI outputs and scientific writing",
                                "Limited reproducibility",
                                "High software costs"
                            ].map((item, i) => (
                                <li key={i} className="flex items-start gap-3">
                                    <div className="mt-1 p-1 rounded-full bg-red-100 text-red-600"><ArrowRight size={12} className="rotate-45" /></div>
                                    <span className="text-slate-600">{item}</span>
                                </li>
                            ))}
                        </ul>
                        <div className="mt-8 p-4 bg-indigo-50 rounded-xl border border-indigo-100 text-indigo-900 font-medium">
                            BioDockify addresses these challenges by modeling pharmaceutical research as an <strong>autonomous reasoning process</strong> rather than a static pipeline.
                        </div>
                    </div>
                </div>
            </section>

            {/* Comparison Table */}
            <section className="py-20 bg-slate-50 border-y border-slate-200">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl lg:text-4xl font-bold mb-4">What Makes BioDockify Different?</h2>
                        <p className="text-slate-600">A fundamental shift from tool-centric to reasoning-centric research.</p>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full bg-white rounded-2xl shadow-sm border-hidden overflow-hidden">
                            <thead className="bg-slate-900 text-white">
                                <tr>
                                    <th className="py-5 px-8 text-left text-lg">Traditional Tools</th>
                                    <th className="py-5 px-8 text-left text-lg bg-indigo-600">BioDockify Pharma Research AI</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {[
                                    ["Manual workflows", "Autonomous agent-driven workflows"],
                                    ["Tool-centric", "Reasoning-centric"],
                                    ["Static analysis", "Iterative self-correcting research"],
                                    ["Black-box AI", "Evidence-constrained AI"],
                                    ["Task-level output", "Thesis- and publication-level output"]
                                ].map(([bad, good], i) => (
                                    <tr key={i} className="hover:bg-slate-50/50">
                                        <td className="py-5 px-8 text-slate-500 font-medium">{bad}</td>
                                        <td className="py-5 px-8 text-indigo-900 font-bold bg-indigo-50/10">{good}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>

            {/* Core Capabilities */}
            <section className="py-20 px-6 max-w-7xl mx-auto">
                <h2 className="text-4xl font-bold text-center mb-16">Core Capabilities</h2>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {/* 1. Autonomous Research Planning */}
                    <div className="p-8 rounded-3xl border border-slate-200 hover:border-indigo-200 hover:shadow-xl hover:shadow-indigo-500/5 transition-all bg-white group">
                        <div className="w-12 h-12 bg-indigo-100 text-indigo-600 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-indigo-600 group-hover:text-white transition-colors">
                            <Brain size={24} />
                        </div>
                        <h3 className="text-xl font-bold mb-3">1. Autonomous Research Planning</h3>
                        <p className="text-slate-600 text-sm mb-4">Powered by Agent Zero, BioDockify acts like a junior researcher.</p>
                        <ul className="space-y-2 text-sm text-slate-500">
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Decomposes goals into tasks</li>
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Selects tools automatically</li>
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Executes in controlled sandbox</li>
                        </ul>
                    </div>

                    {/* 2. Literature Intelligence */}
                    <div className="p-8 rounded-3xl border border-slate-200 hover:border-indigo-200 hover:shadow-xl hover:shadow-indigo-500/5 transition-all bg-white group">
                        <div className="w-12 h-12 bg-blue-100 text-blue-600 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-blue-600 group-hover:text-white transition-colors">
                            <Search size={24} />
                        </div>
                        <h3 className="text-xl font-bold mb-3">2. Evidence-First Literature</h3>
                        <p className="text-slate-600 text-sm mb-4">Systematic analysis of PubMed, OpenAlex, and more.</p>
                        <ul className="space-y-2 text-sm text-slate-500">
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Automated harvesting</li>
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Structure-aware PDF parsing</li>
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Temporal trend analysis</li>
                        </ul>
                    </div>

                    {/* 3. Theme Discovery */}
                    <div className="p-8 rounded-3xl border border-slate-200 hover:border-indigo-200 hover:shadow-xl hover:shadow-indigo-500/5 transition-all bg-white group">
                        <div className="w-12 h-12 bg-purple-100 text-purple-600 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-purple-600 group-hover:text-white transition-colors">
                            <Globe size={24} />
                        </div>
                        <h3 className="text-xl font-bold mb-3">3. Research Theme Discovery</h3>
                        <p className="text-slate-600 text-sm mb-4">Quantify novelty and avoid redundant research.</p>
                        <ul className="space-y-2 text-sm text-slate-500">
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Semantic embedding</li>
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Concept clustering</li>
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Emerging theme detection</li>
                        </ul>
                    </div>

                    {/* 4. Gap & Novelty Analysis */}
                    <div className="p-8 rounded-3xl border border-slate-200 hover:border-indigo-200 hover:shadow-xl hover:shadow-indigo-500/5 transition-all bg-white group">
                        <div className="w-12 h-12 bg-pink-100 text-pink-600 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-pink-600 group-hover:text-white transition-colors">
                            <Layers size={24} />
                        </div>
                        <h3 className="text-xl font-bold mb-3">4. Gap & Novelty Analysis</h3>
                        <p className="text-slate-600 text-sm mb-4">Quantitative scoring for your research proposals.</p>
                        <ul className="space-y-2 text-sm text-slate-500">
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Drug-Target gap detection</li>
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Risk-feasibility assessment</li>
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Perfect for PhD proposals</li>
                        </ul>
                    </div>

                    {/* 5. Knowledge Graph */}
                    <div className="p-8 rounded-3xl border border-slate-200 hover:border-indigo-200 hover:shadow-xl hover:shadow-indigo-500/5 transition-all bg-white group">
                        <div className="w-12 h-12 bg-emerald-100 text-emerald-600 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-emerald-600 group-hover:text-white transition-colors">
                            <Network size={24} />
                        </div>
                        <h3 className="text-xl font-bold mb-3">5. Knowledge Graph Reasoning</h3>
                        <p className="text-slate-600 text-sm mb-4">Links drugs, genes, and diseases to infer mechanisms.</p>
                        <ul className="space-y-2 text-sm text-slate-500">
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Graph-based inference</li>
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Explains results beyond scores</li>
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Hypothesis generation</li>
                        </ul>
                    </div>

                    {/* 6. Thesis-Ready Outputs */}
                    <div className="p-8 rounded-3xl border border-slate-200 hover:border-indigo-200 hover:shadow-xl hover:shadow-indigo-500/5 transition-all bg-white group">
                        <div className="w-12 h-12 bg-amber-100 text-amber-600 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-amber-600 group-hover:text-white transition-colors">
                            <FileText size={24} />
                        </div>
                        <h3 className="text-xl font-bold mb-3">6. Thesis & Publication Ready</h3>
                        <p className="text-slate-600 text-sm mb-4">Controlled AI synthesis traceable to citations.</p>
                        <ul className="space-y-2 text-sm text-slate-500">
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> DOCX & LaTeX support</li>
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> BibTeX reference management</li>
                            <li className="flex gap-2"><CheckCircle size={14} className="text-green-500 shrink-0 mt-0.5" /> Zero hallucination (Evidence-bounded)</li>
                        </ul>
                    </div>
                </div>
            </section>

            {/* Tech Stack & Compliance */}
            <section className="py-20 bg-slate-900 text-white">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
                        <div>
                            <h2 className="text-3xl font-bold mb-8">Technology Stack</h2>
                            <div className="space-y-6">
                                {[
                                    { label: "Agent Orchestration", val: "Agent Zero architecture" },
                                    { label: "Execution Environment", val: "Docker-based sandbox" },
                                    { label: "Databases", val: "Prisma (metadata), Neo4j (graph)" },
                                    { label: "AI Models", val: "Free-tier reasoning APIs (No GPU needed)" },
                                    { label: "Data Sources", val: "PubMed, Europe PMC, OpenAlex, CrossRef" }
                                ].map((item, i) => (
                                    <div key={i} className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800 pb-4">
                                        <span className="text-slate-400 font-medium">{item.label}</span>
                                        <span className="text-indigo-300 font-mono mt-1 sm:mt-0">{item.val}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div>
                            <h2 className="text-3xl font-bold mb-8">Ethical & Academic Compliance</h2>
                            <p className="text-slate-400 leading-relaxed mb-6">
                                BioDockify is designed from the ground up to support academic integrity. The platform assists research but does not replace scientific judgment.
                            </p>
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                {[
                                    "Human-in-the-loop control",
                                    "Transparent reasoning logs",
                                    "Reproducible execution",
                                    "Academic safeguards"
                                ].map((item, i) => (
                                    <div key={i} className="flex items-center gap-3 p-4 bg-slate-800 rounded-xl">
                                        <Shield className="text-emerald-400" size={20} />
                                        <span className="text-sm font-medium">{item}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Final CTA */}
            <section className="py-24 max-w-5xl mx-auto px-6 text-center">
                <h2 className="text-4xl lg:text-5xl font-bold mb-6">Start Your Autonomous Research Journey</h2>
                <p className="text-xl text-slate-500 mb-10 max-w-2xl mx-auto">
                    Transform your pharmaceutical research from a fragmented, manual process into an automated, intellectually rigorous workflow.
                </p>
                <div className="flex flex-col sm:flex-row justify-center items-center gap-6">
                    <a
                        href="https://github.com/tajo9128/BioDockify-pharma-research-ai/releases"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="w-full sm:w-auto px-10 py-5 bg-indigo-600 text-white rounded-full font-bold text-lg hover:bg-indigo-700 transition-all shadow-xl hover:shadow-2xl flex items-center justify-center gap-3"
                    >
                        <Download size={24} />
                        Download BioDockify v3.5
                    </a>
                    <Link to="/contact" className="w-full sm:w-auto px-10 py-5 bg-white text-slate-700 border border-slate-200 rounded-full font-bold text-lg hover:bg-slate-50 transition-all">
                        Contact Sales
                    </Link>
                </div>
                <div className="mt-8 flex items-center justify-center gap-2 text-sm text-slate-400">
                    <Cpu size={14} /> <span>Compatible with Windows, macOS, Linux</span>
                </div>
            </section>
        </div>
    );
};

export default PharmaResearchAIPage;
