import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Boxes, ArrowRight, Trash2, Activity, Zap, ChevronRight } from 'lucide-react';
import { projectService } from '../services/projectService';
import { predictByDisease } from '../services/aiService';

import ToxicityPanel from '../components/Predictions/ToxicityPanel';

export default function DashboardPage() {
    // ...
    // ...
    {/* Main Content */ }
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
            {/* Quick Predict Section */}
            <QuickPredict />

            {/* Toxicity Check Section */}
            <ToxicityPanel />
        </div>

        <div className="mb-8">
            <h1 className="text-3xl font-bold text-slate-900">Your Projects</h1>
            <p className="text-slate-500 mt-2">Manage your QSAR models and drug discovery campaigns.</p>
        </div>


        {loading ? (
            <div className="flex justify-center py-20">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
            </div>
        ) : projects.length === 0 ? (
            <div className="text-center py-20 bg-white rounded-2xl border border-dashed border-slate-300">
                <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4 text-slate-400">
                    <Boxes size={32} />
                </div>
                <h3 className="text-lg font-medium text-slate-900">No projects yet</h3>
                <p className="text-slate-500 max-w-sm mx-auto mt-2 mb-6">Start your first AI-powered drug discovery campaign by creating a project.</p>
                <button
                    onClick={() => setShowModal(true)}
                    className="text-indigo-600 font-medium hover:text-indigo-700 hover:underline"
                >
                    Create your first project
                </button>
            </div>
        ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {projects.map(project => (
                    <div
                        key={project.id}
                        onClick={() => navigate(`/project/${project.id}`)}
                        className="group bg-white rounded-xl border border-slate-200 p-6 hover:shadow-xl hover:shadow-indigo-100 hover:border-indigo-200 transition-all cursor-pointer relative overflow-hidden"
                    >
                        <div className="absolute top-0 right-0 p-4 opacity-0 group-hover:opacity-100 transition-opacity">
                            <button
                                onClick={(e) => handleDelete(project.id, e)}
                                className="text-slate-400 hover:text-red-500 p-1 bg-white rounded shadow-sm"
                            >
                                <Trash2 size={16} />
                            </button>
                        </div>

                        <div className="flex items-start justify-between mb-4">
                            <div className="w-10 h-10 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center">
                                <Activity size={20} />
                            </div>
                            <span className="text-xs font-semibold px-2 py-1 bg-slate-100 text-slate-600 rounded">
                                QSAR
                            </span>
                        </div>

                        <h3 className="text-lg font-bold text-slate-900 mb-2 group-hover:text-indigo-600 transition-colors">
                            {project.name}
                        </h3>
                        <p className="text-sm text-slate-500 line-clamp-2 h-10 mb-6">
                            {project.description || "No description provided."}
                        </p>

                        <div className="flex items-center justify-between text-sm pt-4 border-t border-slate-100">
                            <span className="text-slate-400">
                                {new Date(project.created_at).toLocaleDateString()}
                            </span>
                            <span className="flex items-center text-indigo-600 font-medium opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all">
                                Open <ArrowRight size={14} className="ml-1" />
                            </span>
                        </div>
                    </div>
                ))}
            </div>
        )}
    </main>

    {/* Modal */ }
    {
        showModal && (
            <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                <div className="bg-white rounded-2xl w-full max-w-md p-6 shadow-2xl transform transition-all">
                    <h2 className="text-xl font-bold mb-4">New Project</h2>
                    <form onSubmit={handleCreate}>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-1">Project Name</label>
                                <input
                                    type="text"
                                    required
                                    className="w-full px-4 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
                                    placeholder="e.g. EGFR Inhibitors"
                                    value={newProjectName}
                                    onChange={e => setNewProjectName(e.target.value)}
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-1">Description</label>
                                <textarea
                                    className="w-full px-4 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all resize-none h-24"
                                    placeholder="Briefly describe the goal..."
                                    value={newProjectDesc}
                                    onChange={e => setNewProjectDesc(e.target.value)}
                                />
                            </div>
                        </div>
                        <div className="mt-8 flex justify-end gap-3">
                            <button
                                type="button"
                                onClick={() => setShowModal(false)}
                                className="px-4 py-2 text-slate-600 hover:bg-slate-50 rounded-lg font-medium transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                type="submit"
                                className="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-medium shadow-lg shadow-indigo-200 transition-colors"
                            >
                                Create Project
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        )
    }
        </div >
    );
}

// Quick Predict Component - Allows instant prediction without project creation
function QuickPredict() {
    const [smiles, setSmiles] = useState('');
    const [disease, setDisease] = useState('alzheimers');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const diseases = [
        { id: 'alzheimers', name: "Alzheimer's", gradient: 'from-purple-500 to-indigo-600' },
        { id: 'cancer', name: 'Cancer', gradient: 'from-red-500 to-pink-600' },
        { id: 'diabetes', name: 'Diabetes', gradient: 'from-amber-500 to-orange-600' },
    ];

    const handleQuickPredict = async () => {
        if (!smiles.trim()) return;
        setLoading(true);
        setResult(null);
        try {
            const response = await predictByDisease([smiles], disease);
            if (response.data.predictions?.length > 0) {
                setResult(response.data.predictions[0]);
            }
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const selectedDisease = diseases.find(d => d.id === disease);

    return (
        <div className="mb-10 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-8 text-white relative overflow-hidden">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMTAiIGN5PSIxMCIgcj0iMiIgZmlsbD0icmdiYSgyNTUsMjU1LDI1NSwwLjEpIi8+PC9zdmc+')] opacity-50"></div>
            <div className="relative z-10">
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center">
                        <Zap size={20} />
                    </div>
                    <div>
                        <h2 className="text-xl font-bold">Quick Predict</h2>
                        <p className="text-white/70 text-sm">Test a molecule instantly with AI</p>
                    </div>
                </div>

                <div className="flex flex-wrap gap-2 mb-4">
                    {diseases.map(d => (
                        <button
                            key={d.id}
                            onClick={() => setDisease(d.id)}
                            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${disease === d.id
                                ? 'bg-white text-indigo-600'
                                : 'bg-white/20 text-white hover:bg-white/30'
                                }`}
                        >
                            {d.name}
                        </button>
                    ))}
                </div>

                <div className="flex gap-3">
                    <input
                        type="text"
                        value={smiles}
                        onChange={(e) => setSmiles(e.target.value)}
                        placeholder="Enter SMILES string (e.g., CC(C)Cc1ccc(cc1)C(C)C(=O)O)"
                        className="flex-1 px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder:text-white/50 focus:outline-none focus:ring-2 focus:ring-white/50"
                    />
                    <button
                        onClick={handleQuickPredict}
                        disabled={loading || !smiles.trim()}
                        className="px-6 py-3 bg-white text-indigo-600 font-bold rounded-xl hover:bg-indigo-50 transition-all flex items-center gap-2 disabled:opacity-50"
                    >
                        {loading ? 'Predicting...' : 'Predict'} <ChevronRight size={18} />
                    </button>
                </div>

                {result && (
                    <div className={`mt-4 p-4 rounded-xl ${result.prediction === 'Active' ? 'bg-green-500/30' :
                        result.prediction === 'Moderate' ? 'bg-yellow-500/30' :
                            'bg-red-500/30'
                        }`}>
                        <div className="flex justify-between items-center">
                            <span className="font-bold text-lg">{result.prediction}</span>
                            <span className="text-2xl font-bold">{(result.score * 100).toFixed(1)}%</span>
                        </div>
                        {result.interpretation && (
                            <p className="text-sm text-white/80 mt-2">{result.interpretation}</p>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
