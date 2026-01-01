import React from 'react';
import { Link } from 'react-router-dom';

const tools = [
    {
        id: 'converter',
        title: 'SMILES to PDBQT',
        description: 'Convert chemical SMILES strings into 3D PDBQT format ready for docking. Includes automatic 3D conformation generation.',
        icon: '🔄',
        path: '/tools/converter',
        color: 'bg-blue-50 text-blue-600',
        btnColor: 'bg-blue-600 hover:bg-blue-700'
    },
    {
        id: 'uniprot',
        title: 'UniProt Target Explorer',
        description: 'Fetch detailed protein target metadata, including gene names, function descriptions, and active site locations directly from UniProt.',
        icon: '🧬',
        path: '/tools/target-explorer',
        color: 'bg-emerald-50 text-emerald-600',
        btnColor: 'bg-emerald-600 hover:bg-emerald-700'
    },
    {
        id: 'rcsb',
        title: 'RCSB Structure Checker',
        description: 'Validate PDB structures before docking. Check resolution, experimental method, and release dates to ensure quality.',
        icon: '🏗️',
        path: '/tools/structure-search',
        color: 'bg-violet-50 text-violet-600',
        btnColor: 'bg-violet-600 hover:bg-violet-700'
    },
    {
        id: 'chembl',
        title: 'ChEMBL Bioactivity',
        description: 'Lookup experimental IC50, Ki, and KD values for known inhibitors. Benchmark your docking scores against real-world data.',
        icon: '💊',
        path: '/tools/bioactivity',
        color: 'bg-rose-50 text-rose-600',
        btnColor: 'bg-rose-600 hover:bg-rose-700'
    },
    {
        id: 'prediction',
        title: 'Target Prediction',
        description: 'Predict probable protein targets for your ligand using AI-driven similarity search against ChEMBL database.',
        icon: '🎯',
        path: '/tools/prediction',
        color: 'bg-indigo-50 text-indigo-600',
        btnColor: 'bg-indigo-600 hover:bg-indigo-700'
    },
    {
        id: 'admet',
        title: 'ADMET Predictor',
        description: 'Evaluate Absorption, Distribution, Metabolism, Excretion, and Toxicity properties to filter lead candidates.',
        icon: '🧪',
        path: '/tools/admet',
        color: 'bg-teal-50 text-teal-600',
        btnColor: 'bg-teal-600 hover:bg-teal-700'
    },
    {
        id: 'benchmark',
        title: 'Docking Benchmark',
        description: 'Test the accuracy of BioDockify against standard datasets. Validate your parameters before large screens.',
        icon: '📊',
        path: '/tools/benchmark',
        color: 'bg-cyan-50 text-cyan-600',
        btnColor: 'bg-cyan-600 hover:bg-cyan-700'
    },
    {
        id: '3dviewer',
        title: '3D Structure Viewer',
        description: 'Interactive Mol* viewer for PDB, PDBQT, and SDF files. visualize interactions and binding poses.',
        icon: '🧊',
        path: '/3d-viewer',
        color: 'bg-sky-50 text-sky-600',
        btnColor: 'bg-sky-600 hover:bg-sky-700'
    },
    {
        id: 'pockets',
        title: 'Pocket Detector',
        description: 'Analyze protein geometry to find binding pockets. Returns coordinates and box sizes for grid generation.',
        icon: '🔍',
        path: '/tools/pockets',
        color: 'bg-orange-50 text-orange-600',
        btnColor: 'bg-orange-600 hover:bg-orange-700'
    },
    {
        id: 'prioritization',
        title: 'Lead Prioritization',
        description: 'Portfolio optimization engine using Decision Science. Select the best compounds based on budget ($) and risk tolerance.',
        icon: '⚖️',
        path: '/tools/prioritization',
        color: 'bg-pink-50 text-pink-600',
        btnColor: 'bg-pink-600 hover:bg-pink-700'
    },
    {
        id: 'developability',
        title: 'Developability Warning',
        description: 'Fail-early detector. Flag toxic or insoluble compounds (PAINS, Lipinski violations) before you waste resources.',
        icon: '⚠️',
        path: '/tools/developability',
        color: 'bg-amber-50 text-amber-600',
        btnColor: 'bg-amber-600 hover:bg-amber-700'
    }
];

export default function ToolsPage() {
    return (
        <div className="min-h-screen bg-slate-50 pt-24 pb-12">
            <div className="container mx-auto px-4">

                <div className="text-center max-w-3xl mx-auto mb-16">
                    <h1 className="text-4xl font-extrabold text-slate-900 mb-4">
                        BioDockify <span className="text-primary-600">Smart Tools</span>
                    </h1>
                    <p className="text-lg text-slate-600">
                        A suite of intelligent utilities to accelerate your drug discovery workflow.
                        From data fetching to geometry analysis, we've got you covered.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
                    {tools.map((tool) => (
                        <div key={tool.id} className="bg-white rounded-2xl p-8 shadow-sm hover:shadow-xl transition-all duration-300 border border-slate-100 group">
                            <div className={`w-14 h-14 rounded-xl flex items-center justify-center text-3xl mb-6 ${tool.color} group-hover:scale-110 transition-transform duration-300`}>
                                {tool.icon}
                            </div>

                            <h3 className="text-xl font-bold text-slate-900 mb-3 group-hover:text-primary-600 transition-colors">
                                {tool.title}
                            </h3>

                            <p className="text-slate-600 mb-8 leading-relaxed">
                                {tool.description}
                            </p>

                            <Link
                                to={tool.path}
                                className={`inline-flex items-center justify-center w-full py-3 px-6 rounded-xl text-white font-bold transition-all shadow-md hover:shadow-lg transform active:scale-95 ${tool.btnColor}`}
                            >
                                Launch Tool
                                <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                                </svg>
                            </Link>
                        </div>
                    ))}
                </div>

                {/* Banner for BioDockify AI Agent */}
                <div className="mt-20 bg-gradient-to-r from-slate-900 to-slate-800 rounded-3xl p-10 md:p-16 text-center shadow-2xl relative overflow-hidden">
                    <div className="absolute top-0 right-0 -mr-20 -mt-20 w-80 h-80 bg-primary-600 rounded-full blur-[100px] opacity-20 animate-pulse"></div>
                    <div className="absolute bottom-0 left-0 -ml-20 -mb-20 w-80 h-80 bg-purple-600 rounded-full blur-[100px] opacity-20 animate-pulse"></div>

                    <div className="relative z-10 max-w-2xl mx-auto">
                        <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
                            Powered by BioDockify AI Agent
                        </h2>
                        <p className="text-slate-300 text-lg mb-8">
                            These tools are also integrated directly into our AI assistant.
                            You can ask BioDockify AI Agent to run these analysis steps for you in plain English!
                        </p>
                        <button
                            onClick={() => window.dispatchEvent(new CustomEvent('agent-zero-trigger', { detail: { prompt: "What tools can you use?" } }))}
                            className="bg-white text-slate-900 px-8 py-3 rounded-full font-bold hover:bg-primary-50 transition-colors shadow-lg hover:shadow-xl"
                        >
                            Ask BioDockify AI Agent
                        </button>
                    </div>
                </div>

            </div>
        </div>
    );
}
