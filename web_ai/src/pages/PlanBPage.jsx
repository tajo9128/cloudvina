import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function PlanBPage() {
    const navigate = useNavigate();
    const [plant, setPlant] = useState('');
    const [disease, setDisease] = useState('');

    const handleUpload = () => {
        // Logic to trigger upload
        console.log("Triggering Plan B Upload for:", { plant, disease });
        // Could open file picker here programmatically or navigate
        navigate('/dashboard');
    };

    return (
        <div className="max-w-7xl mx-auto px-4 py-12">
            <div className="mb-8">
                <button onClick={() => navigate('/')} className="text-slate-500 hover:text-primary-600 mb-4 flex items-center gap-1 text-sm font-medium">
                    ← Back to Plans
                </button>
                <div className="flex items-center gap-3 mb-2">
                    <span className="bg-primary-600 text-white px-3 py-1 rounded-md text-sm font-bold">PLAN B</span>
                    <h1 className="text-3xl font-bold text-slate-900">Plant-First Phytochemical Discovery</h1>
                </div>
                <p className="text-slate-600 text-lg max-w-3xl">
                    Execute **Phases 1-3** of the BioDockify research framework. Upload GC-MS data to identify phytochemicals and predict their bioactivity against CNS targets using AI.
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 space-y-8">
                    {/* Step 1: Plant Source */}
                    <div className="bg-white border boundary-slate-200 rounded-xl p-8 shadow-sm">
                        <h2 className="text-xl font-semibold text-slate-900 mb-6">1. Plant Source Configuration</h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-2">Plant Species</label>
                                <select
                                    className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all"
                                    value={plant}
                                    onChange={(e) => setPlant(e.target.value)}
                                >
                                    <option value="">Select Plant...</option>
                                    <option value="evolvulus">Evolvulus alsinoides (Dwarf Morning Glory)</option>
                                    <option value="cordia">Cordia dichotoma (Lasura)</option>
                                    <option value="withania">Withania somnifera (Ashwagandha)</option>
                                    <option value="custom">Custom / Other</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-slate-700 mb-2">Target Indication</label>
                                <select
                                    className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all"
                                    value={disease}
                                    onChange={(e) => setDisease(e.target.value)}
                                >
                                    <option value="">Select Disease...</option>
                                    <option value="alzheimers">Alzheimer's Disease</option>
                                    <option value="general">General CNS Neuroprotection</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Step 2: Data Upload */}
                    <div className="bg-white border boundary-slate-200 rounded-xl p-8 shadow-sm">
                        <h2 className="text-xl font-semibold text-slate-900 mb-4">2. GC-MS Data Upload</h2>
                        <p className="text-sm text-slate-500 mb-6">
                            Upload your NetCDF (.cdf) or processed CSV files containing peak lists (Retention Time, m/z, Area).
                        </p>

                        <div className="border-2 border-dashed border-slate-300 rounded-xl p-10 text-center hover:border-primary-500 hover:bg-primary-50 transition-all cursor-pointer group">
                            <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">📂</div>
                            <h3 className="font-semibold text-slate-900">Click to upload raw chromatogram data</h3>
                            <p className="text-sm text-slate-500 mt-2">Max file size: 50MB</p>
                        </div>
                    </div>

                    {/* Action */}
                    <button
                        onClick={handleUpload}
                        disabled={!plant}
                        className="w-full bg-primary-600 text-white py-4 rounded-xl text-lg font-bold hover:bg-primary-700 transition-colors shadow-lg shadow-primary-600/20 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        📤 Upload & Identify Phytochemicals
                    </button>
                </div>

                {/* Sidebar Info */}
                <div className="space-y-6">
                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-6">
                        <h3 className="font-semibold text-slate-900 mb-3">Phase Workflow</h3>
                        <div className="space-y-4">
                            <div className="relative pl-6 border-l-2 border-slate-300 pb-4">
                                <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-slate-300"></div>
                                <strong className="block text-sm text-slate-900">Phase 1: Extraction</strong>
                                <span className="text-xs text-slate-500">GC-MS signal processing & peak detection</span>
                            </div>
                            <div className="relative pl-6 border-l-2 border-primary-500 pb-4">
                                <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-primary-500 ring-4 ring-primary-100"></div>
                                <strong className="block text-sm text-slate-900">Phase 2: AI ID</strong>
                                <span className="text-xs text-slate-500">ChemBERTa inference for compound ID</span>
                            </div>
                            <div className="relative pl-6 border-l-2 border-slate-300">
                                <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-slate-300"></div>
                                <strong className="block text-sm text-slate-900">Phase 3: SAR</strong>
                                <span className="text-xs text-slate-500">Bioactivity prediction on targets</span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-6 text-sm text-indigo-800">
                        <strong>🎓 PhD Output:</strong> This workflow automatically generates Tables & Figures suitable for "Chapter 3: Phytochemical Screening" of your thesis.
                    </div>
                </div>
            </div>
        </div>
    );
}
