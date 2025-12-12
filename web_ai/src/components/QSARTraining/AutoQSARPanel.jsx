import React, { useState } from 'react';
import { aiService } from '../../services/aiService';

const AutoQSARPanel = () => {
    const [file, setFile] = useState(null);
    const [targetCol, setTargetCol] = useState('pIC50');
    const [modelName, setModelName] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setError(null);
    };

    const handleTrain = async () => {
        if (!file || !modelName) {
            setError("Please select a file and provide a model name.");
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const formData = new FormData();
            formData.append('file', file);

            // Note: These query params need to match the backend endpoint
            const response = await aiService.trainAutoQSAR(formData, targetCol, modelName);
            setResult(response);
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || "Training failed. Please check your CSV format.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h2 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                <span className="text-2xl">🤖</span> Auto-QSAR Builder
            </h2>

            <div className="space-y-4">
                {/* Input Section */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-slate-600 mb-1">Dataset (CSV)</label>
                        <input
                            type="file"
                            accept=".csv"
                            onChange={handleFileChange}
                            className="w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
                        />
                        <p className="text-xs text-slate-400 mt-1">Must contain 'smiles' and target column. Max 1000 rows.</p>
                    </div>

                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm font-medium text-slate-600 mb-1">Target Column Name</label>
                            <input
                                type="text"
                                value={targetCol}
                                onChange={(e) => setTargetCol(e.target.value)}
                                className="w-full p-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
                                placeholder="e.g., pIC50, activity"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-600 mb-1">Model Name</label>
                            <input
                                type="text"
                                value={modelName}
                                onChange={(e) => setModelName(e.target.value)}
                                className="w-full p-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
                                placeholder="e.g., My_Alzheimers_Model"
                            />
                        </div>
                    </div>
                </div>

                {/* Train Button */}
                <button
                    onClick={handleTrain}
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-primary-600 to-indigo-600 text-white font-semibold py-3 rounded-lg hover:shadow-lg transition-all disabled:opacity-70 disabled:cursor-not-allowed flex justify-center items-center gap-2"
                >
                    {loading ? (
                        <>
                            <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Training Random Forest...
                        </>
                    ) : (
                        "Train Model"
                    )}
                </button>

                {error && (
                    <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm border border-red-200">
                        Error: {error}
                    </div>
                )}
            </div>

            {/* Results Section */}
            {result && (
                <div className="mt-6 border-t border-slate-100 pt-6 animate-fadeIn">
                    <h3 className="text-lg font-semibold text-slate-800 mb-3">Training Results</h3>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                        <div className="bg-green-50 p-4 rounded-lg border border-green-100 text-center">
                            <div className="text-2xl font-bold text-green-700">{result.metrics.test_r2.toFixed(3)}</div>
                            <div className="text-xs text-green-600 uppercase tracking-wide font-semibold">Test R² Score</div>
                        </div>
                        <div className="bg-blue-50 p-4 rounded-lg border border-blue-100 text-center">
                            <div className="text-2xl font-bold text-blue-700">{result.metrics.rmse.toFixed(3)}</div>
                            <div className="text-xs text-blue-600 uppercase tracking-wide font-semibold">RMSE</div>
                        </div>
                        <div className="bg-purple-50 p-4 rounded-lg border border-purple-100 text-center">
                            <div className="text-2xl font-bold text-purple-700">{result.metrics.cv_r2.toFixed(3)}</div>
                            <div className="text-xs text-purple-600 uppercase tracking-wide font-semibold">CV Score (3-Fold)</div>
                        </div>
                    </div>

                    <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
                        <h4 className="font-medium text-slate-700 mb-2">Dataset Info</h4>
                        <div className="text-sm text-slate-600">
                            <p>Rows Processed: <span className="font-mono text-slate-900">{result.dataset_size}</span></p>
                            <p>Model ID: <span className="font-mono text-slate-900">{result.model_id}</span></p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AutoQSARPanel;
