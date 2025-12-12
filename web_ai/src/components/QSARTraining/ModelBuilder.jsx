import React, { useState } from 'react';
import { Beaker, CheckCircle } from 'lucide-react';

export default function ModelBuilder({ compounds, onTrain, trainStatus }) {
    const [config, setConfig] = useState({ name: '', target: 'activity' });

    const handleTrainStart = () => {
        onTrain(config);
    };

    return (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
            <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Model Name</label>
                <input
                    className="w-full px-4 py-2 rounded-lg border border-slate-300 focus:border-indigo-500 outline-none"
                    placeholder="e.g. Solubility Predictor v1"
                    value={config.name}
                    onChange={e => setConfig({ ...config, name: e.target.value })}
                />
            </div>

            <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Target Column</label>
                <select
                    className="w-full px-4 py-2 rounded-lg border border-slate-300 focus:border-indigo-500 outline-none"
                    value={config.target}
                    onChange={e => setConfig({ ...config, target: e.target.value })}
                >
                    <option value="">Select a column...</option>
                    {compounds.length > 0 && Object.keys(compounds[0].properties)
                        .filter(k => k !== 'smiles' && k !== 'name')
                        .map(k => <option key={k} value={k}>{k}</option>)
                    }
                    {/* Fallback if no data */}
                    <option value="activity">activity (default)</option>
                    <option value="logP">logP</option>
                </select>
            </div>

            <div className="p-4 bg-indigo-50 rounded-lg border border-indigo-100 flex gap-3">
                <Beaker className="text-indigo-600 flex-shrink-0" />
                <div className="text-sm text-indigo-900">
                    <p className="font-semibold">ChemBERTa Engine</p>
                    <p>Models will be trained using Hugging Face Spaces (Free Tier). Large datasets may take time.</p>
                </div>
            </div>

            <button
                onClick={handleTrainStart}
                disabled={trainStatus === 'training'}
                className={`w-full py-3 rounded-lg font-bold text-white transition-all flex justify-center items-center gap-2
                    ${trainStatus === 'training' ? 'bg-indigo-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 shadow-lg shadow-indigo-200'}
                `}
            >
                {trainStatus === 'training' ? (
                    <><div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full" /> Training...</>
                ) : 'Start Training'}
            </button>

            {trainStatus === 'success' && (
                <div className="flex items-center gap-2 text-green-600 bg-green-50 p-3 rounded-lg border border-green-100">
                    <CheckCircle size={18} /> Training initiated! Check Models tab for results.
                </div>
            )}
        </div>
    );
}
