import React, { useState } from 'react';
import { Database, Brain, network, Settings, Zap, ArrowRight, Upload } from 'lucide-react';

function App() {
    const [activePhase, setActivePhase] = useState(1);

    return (
        <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-ai-500 selection:text-white">
            {/* Header */}
            <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
                <div className="container mx-auto px-6 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-gradient-to-br from-ai-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-ai-500/20">
                            <Brain className="w-6 h-6 text-white" />
                        </div>
                        <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                            BioDockify <span className="text-ai-500">AI</span>
                        </span>
                    </div>
                    <nav className="hidden md:flex gap-6 text-sm font-medium text-slate-400">
                        <a href="#" className="hover:text-white transition-colors">Documentation</a>
                        <a href="#" className="hover:text-white transition-colors">Models</a>
                        <a href="#" className="text-ai-500">Zero-Cost Roadmap</a>
                    </nav>
                </div>
            </header>

            {/* Hero Section */}
            <section className="relative py-20 overflow-hidden">
                <div className="absolute inset-0 bg-grid-slate-900/[0.04] bg-[bottom_1px_center] pointer-events-none" />
                <div className="container mx-auto px-6 text-center relative z-10">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-ai-500/10 text-ai-400 text-xs font-medium border border-ai-500/20 mb-8">
                        <Zap className="w-3 h-3" /> Powered by Zero-Cost Infrastructure
                    </div>
                    <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6">
                        Zero-Cost AI Drug Discovery <br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-ai-400 to-indigo-400">
                            Enterprise Quality. Free Forever.
                        </span>
                    </h1>
                    <p className="text-slate-400 max-w-2xl mx-auto text-lg mb-10 leading-relaxed">
                        Execute the 16-Week Roadmap: From Data Foundation to Deep Learning & Automation.
                        Run state-of-the-art models without spending a dollar.
                    </p>

                    <div className="flex justify-center gap-4">
                        <button className="px-6 py-3 bg-ai-600 hover:bg-ai-500 text-white rounded-lg font-semibold transition-all shadow-lg shadow-ai-500/25 flex items-center gap-2">
                            Start Phase {activePhase} <ArrowRight className="w-4 h-4" />
                        </button>
                        <button className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg font-medium transition-all border border-slate-700">
                            View Architecture
                        </button>
                    </div>
                </div>
            </section>

            {/* Roadmap Visualization */}
            <section className="container mx-auto px-6 py-16">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-12">
                    {/* Phase Cards */}
                    <PhaseCard
                        step={1}
                        title="Data Foundation"
                        weeks="Weeks 1-2"
                        icon={<Database />}
                        active={activePhase === 1}
                        onClick={() => setActivePhase(1)}
                    />
                    <PhaseCard
                        step={2}
                        title="Classical ML"
                        weeks="Weeks 3-6"
                        icon={<Brain />}
                        active={activePhase === 2}
                        onClick={() => setActivePhase(2)}
                    />
                    <PhaseCard
                        step={3}
                        title="Deep Learning"
                        weeks="Weeks 7-12"
                        icon={<Zap />}
                        active={activePhase === 3}
                        onClick={() => setActivePhase(3)}
                    />
                    <PhaseCard
                        step={4}
                        title="Automation"
                        weeks="Weeks 13-16"
                        icon={<Settings />}
                        active={activePhase === 4}
                        onClick={() => setActivePhase(4)}
                    />
                </div>

                {/* Active Phase Detail */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl">
                    {activePhase === 1 && <Phase1Detail />}
                    {activePhase === 2 && <Phase2Detail />}
                    {activePhase === 3 && <Phase3Detail />}
                    {activePhase === 4 && <Phase4Detail />}
                </div>
            </section>
        </div>
    );
}

// --- Components ---

function PhaseCard({ step, title, weeks, icon, active, onClick }) {
    return (
        <div
            onClick={onClick}
            className={`cursor-pointer p-6 rounded-xl border transition-all duration-300 group
      ${active
                    ? 'bg-ai-500/10 border-ai-500/50 shadow-lg shadow-ai-500/10'
                    : 'bg-slate-800/50 border-slate-700 hover:border-slate-600 hover:bg-slate-800'
                }`}
        >
            <div className={`mb-4 w-10 h-10 rounded-lg flex items-center justify-center
        ${active ? 'bg-ai-500 text-white' : 'bg-slate-700 text-slate-400 group-hover:text-white'}`}>
                {React.cloneElement(icon, { size: 20 })}
            </div>
            <div className="text-xs font-mono text-slate-500 mb-1">{weeks}</div>
            <h3 className={`font-semibold ${active ? 'text-white' : 'text-slate-300'}`}>Phase {step}: {title}</h3>
        </div>
    )
}

function Phase1Detail() {
    return (
        <div className="flex flex-col md:flex-row gap-12 items-start">
            <div className="flex-1">
                <h2 className="text-2xl font-bold mb-4 flex items-center gap-3">
                    <Database className="text-ai-500" /> Data Foundation
                </h2>
                <p className="text-slate-400 mb-6">
                    Centralize your experimental data and computational predictions into a robust SQLite/PostgreSQL database.
                </p>
                <div className="space-y-4">
                    <CheckItem text="Create SQLite Database Schema" done />
                    <CheckItem text="Compile Experimental CSVs" />
                    <CheckItem text="Standardize SMILES with RDKit" />
                    <CheckItem text="Data Quality Report" />
                </div>
            </div>
            <div className="flex-1 bg-slate-950 rounded-xl p-6 border border-slate-800 w-full">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="font-medium text-slate-300">Data Ingestion</h3>
                    <span className="text-xs bg-slate-800 px-2 py-1 rounded text-slate-400">RDKit Ready</span>
                </div>
                <div className="border-2 border-dashed border-slate-800 rounded-lg h-32 flex flex-col items-center justify-center text-slate-500 hover:border-ai-500/50 hover:bg-slate-900 transition-colors cursor-pointer">
                    <Upload className="mb-2 w-8 h-8 opacity-50" />
                    <span className="text-sm">Drop CSV files here</span>
                </div>
            </div>
        </div>
    )
}

function Phase2Detail() {
    return (
        <div className="space-y-6">
            <h2 className="text-2xl font-bold flex items-center gap-3">
                <Brain className="text-ai-500" /> Classical ML Models
            </h2>
            <div className="grid grid-cols-2 gap-4">
                <ModelCard name="QSAR IC50" type="Random Forest" acc="0.82 R²" status="Ready" />
                <ModelCard name="BBB Penetration" type="Classifier" acc="94%" status="Training" />
                <ModelCard name="Toxicity" type="Ensemble" acc="89%" status="Pending" />
                <ModelCard name="NP-Likeness" type="Heuristic" acc="N/A" status="Active" />
            </div>
        </div>
    )
}

function Phase3Detail() {
    return (
        <div className="text-center py-12">
            <Zap className="w-16 h-16 text-slate-700 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-slate-300">Deep Learning Phase</h2>
            <p className="text-slate-500 mt-2">Unlocks in Weeks 7-12 via Google Colab & Hugging Face.</p>
        </div>
    )
}

function Phase4Detail() {
    return (
        <div className="text-center py-12">
            <Settings className="w-16 h-16 text-slate-700 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-slate-300">Automation Phase</h2>
            <p className="text-slate-500 mt-2">Active Learning Loops unlocking in Weeks 13-16.</p>
        </div>
    )
}


function CheckItem({ text, done }) {
    return (
        <div className="flex items-center gap-3 text-slate-300">
            <div className={`w-5 h-5 rounded border flex items-center justify-center ${done ? 'bg-green-500 border-green-500' : 'border-slate-600'}`}>
                {done && <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" /></svg>}
            </div>
            <span>{text}</span>
        </div>
    )
}

function ModelCard({ name, type, acc, status }) {
    return (
        <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 flex justify-between items-center">
            <div>
                <h4 className="font-semibold text-slate-200">{name}</h4>
                <div className="text-xs text-slate-500">{type}</div>
            </div>
            <div className="text-right">
                <div className="text-sm font-mono text-ai-400">{acc}</div>
                <div className="text-xs text-slate-600">{status}</div>
            </div>
        </div>
    )
}

export default App;
