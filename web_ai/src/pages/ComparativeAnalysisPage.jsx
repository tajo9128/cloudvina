import React, { useState } from 'react';
import { ArrowLeft, Upload, Activity, Zap, BarChart2, FileText, CheckCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

// Mock AI Service for Simulation
const mockAnalyze = async (groupASmiles, groupBSmiles) => {
    // Simulate API delay
    await new Promise(r => setTimeout(r, 2000));

    // Generate Mock pIC50 scores (Gaussian)
    const generateScores = (count, mean) => Array.from({ length: count }, () => mean + (Math.random() - 0.5) * 2);

    // Evolvulus (Plant A) assumes slightly better distribution
    const scoresA = generateScores(groupASmiles.split('\n').filter(s => s.trim()).length || 10, 7.2);
    // Cordia (Plant B) assumes slightly lower
    const scoresB = generateScores(groupBSmiles.split('\n').filter(s => s.trim()).length || 10, 6.5);

    return { scoresA, scoresB };
};

export default function ComparativeAnalysisPage() {
    const navigate = useNavigate();
    const [activeStep, setActiveStep] = useState(0); // 0: Input, 1: Analyzing, 2: Results

    // Inputs
    const [plantAName, setPlantAName] = useState("Evolvulus alsinoides");
    const [plantASmiles, setPlantASmiles] = useState("CC1=CC(=O)C2=C(C1=O)C(=CC=C2O)O\nC1=CC(=C(C(=C1)O)O)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O");

    const [plantBName, setPlantBName] = useState("Cordia dichotoma");
    const [plantBSmiles, setPlantBSmiles] = useState("CC1=CC(=O)C2=CC(=C(C=C2C1=O)O)O\nCOC1=C(C=CC(=C1)O)C(CC(=O)C2=CC(=C(C=C2)O)O)O");

    // Results
    const [results, setResults] = useState(null);

    const handleRunAnalysis = async () => {
        setActiveStep(1);
        const data = await mockAnalyze(plantASmiles, plantBSmiles);
        setResults(data);
        setActiveStep(2);
    };

    return (
        <div className="min-h-screen bg-slate-50 flex flex-col font-sans text-slate-900">
            {/* Header */}
            <header className="bg-white border-b border-slate-200 h-16 flex items-center px-6 sticky top-0 z-20">
                <button onClick={() => navigate('/dashboard')} className="mr-4 p-2 hover:bg-slate-100 rounded-lg text-slate-500">
                    <ArrowLeft size={20} />
                </button>
                <div>
                    <h1 className="text-lg font-bold bg-gradient-to-r from-indigo-600 to-violet-600 bg-clip-text text-transparent">
                        Comparative Phytochemical Profiling
                    </h1>
                    <span className="text-xs text-slate-500 uppercase tracking-wider">AI-Integrated CADD Workbench</span>
                </div>
            </header>

            <main className="flex-1 p-8 max-w-6xl mx-auto w-full">

                {/* STAGE 1: INPUT */}
                {activeStep === 0 && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                        className="grid grid-cols-1 md:grid-cols-2 gap-8"
                    >
                        {/* Plant A Input */}
                        <PlantInputCard
                            title="Plant Group A"
                            name={plantAName} setName={setPlantAName}
                            smiles={plantASmiles} setSmiles={setPlantASmiles}
                            color="indigo"
                        />
                        {/* Plant B Input */}
                        <PlantInputCard
                            title="Plant Group B"
                            name={plantBName} setName={setPlantBName}
                            smiles={plantBSmiles} setSmiles={setPlantBSmiles}
                            color="emerald"
                        />

                        {/* Action Button */}
                        <div className="md:col-span-2 flex justify-center mt-8">
                            <button
                                onClick={handleRunAnalysis}
                                className="flex items-center gap-3 bg-slate-900 text-white px-8 py-4 rounded-xl text-lg font-bold shadow-lg hover:shadow-xl hover:scale-105 transition-all"
                            >
                                <Zap className="text-yellow-400" fill="currentColor" />
                                Run Comparative AI Screen
                            </button>
                        </div>
                    </motion.div>
                )}

                {/* STAGE 2: LOADING */}
                {activeStep === 1 && (
                    <div className="flex flex-col items-center justify-center h-96">
                        <Activity className="w-16 h-16 text-indigo-600 animate-pulse mb-6" />
                        <h2 className="text-2xl font-bold text-slate-800">Profiling Phytochemicals...</h2>
                        <p className="text-slate-500 mt-2">Running Tier 2 Ensemble Model (MolFormer + ChemBERTa)</p>
                        <div className="mt-8 flex gap-4 text-sm text-slate-400">
                            <span>Processing SMILES</span>
                            <span>•</span>
                            <span>Calculating Similarities</span>
                            <span>•</span>
                            <span>Predicting pIC50</span>
                        </div>
                    </div>
                )}

                {/* STAGE 3: RESULTS */}
                {activeStep === 2 && results && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                        <div className="flex justify-between items-center mb-8">
                            <h2 className="text-3xl font-bold text-slate-800">Comparative Profile Analysis</h2>
                            <button onClick={() => setActiveStep(0)} className="text-slate-500 hover:text-indigo-600 font-medium">
                                Start New Analysis
                            </button>
                        </div>

                        {/* Top Stats Cards */}
                        <div className="grid grid-cols-3 gap-6 mb-12">
                            <StatCard
                                label={`Best ${plantAName} Hit`}
                                value={Math.max(...results.scoresA).toFixed(2)}
                                sub="pIC50 Affinity"
                                color="text-indigo-600"
                            />
                            <StatCard
                                label={`Best ${plantBName} Hit`}
                                value={Math.max(...results.scoresB).toFixed(2)}
                                sub="pIC50 Affinity"
                                color="text-emerald-600"
                            />
                            <StatCard
                                label="Winning Plant"
                                value={Math.max(...results.scoresA) > Math.max(...results.scoresB) ? plantAName : plantBName}
                                sub="Based on Max Affinity"
                                color="text-amber-600"
                            />
                        </div>

                        {/* Distribution Chart (Simple Visual) */}
                        <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200 mb-8">
                            <h3 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
                                <BarChart2 size={20} />
                                Bio-Activity Distribution (Predicted pIC50)
                            </h3>
                            <div className="h-64 flex items-end justify-center gap-4 relative">
                                {/* Visualizing simple bars for now - can use Recharts later */}
                                <DistributionBar label="A" scores={results.scoresA} color="bg-indigo-500" name={plantAName} />
                                <DistributionBar label="B" scores={results.scoresB} color="bg-emerald-500" name={plantBName} />
                            </div>
                        </div>
                    </motion.div>
                )}

            </main>
        </div>
    );
}

// Subcomponents

function PlantInputCard({ title, name, setName, smiles, setSmiles, color }) {
    const colorClasses = {
        indigo: "border-indigo-100 bg-indigo-50/50 focus-within:border-indigo-500",
        emerald: "border-emerald-100 bg-emerald-50/50 focus-within:border-emerald-500"
    };

    return (
        <div className={`bg-white p-6 rounded-2xl border-2 ${colorClasses[color]} transition-colors`}>
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold text-slate-700 flex items-center gap-2">
                    <FileText size={18} /> {title}
                </h3>
            </div>
            <div className="space-y-4">
                <div>
                    <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Botanical Name</label>
                    <input
                        type="text"
                        value={name} onChange={e => setName(e.target.value)}
                        className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm font-medium focus:outline-none"
                    />
                </div>
                <div>
                    <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
                        Constituents (SMILES List)
                    </label>
                    <textarea
                        value={smiles} onChange={e => setSmiles(e.target.value)}
                        className="w-full h-48 bg-white border border-slate-200 rounded-lg px-3 py-2 text-xs font-mono focus:outline-none resize-none"
                        placeholder="Paste SMILES here..."
                    />
                </div>
            </div>
        </div>
    );
}

function StatCard({ label, value, sub, color }) {
    return (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <h4 className="text-slate-500 text-sm font-medium uppercase tracking-wider mb-1">{label}</h4>
            <div className={`text-3xl font-black ${color} mb-1`}>{value}</div>
            <div className="text-slate-400 text-xs">{sub}</div>
        </div>
    );
}

function DistributionBar({ scores, color, name }) {
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    // Map 0-10 scale to 0-100% height
    const height = Math.min(100, Math.max(10, (avg / 10) * 100));

    return (
        <div className="flex flex-col items-center w-32 group">
            <motion.div
                initial={{ height: 0 }} animate={{ height: `${height}%` }}
                className={`w-full ${color} rounded-t-xl opacity-80 group-hover:opacity-100 transition-opacity relative`}
            >
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-xs px-2 py-1 rounded">
                    Avg: {avg.toFixed(2)}
                </div>
            </motion.div>
            <div className="mt-3 text-center">
                <div className="font-bold text-slate-700">{name}</div>
                <div className="text-xs text-slate-500">{scores.length} compounds</div>
            </div>
        </div>
    );
}
