import React, { useState, useRef } from 'react';
import { Upload, FileSpreadsheet, Download, AlertCircle, CheckCircle } from 'lucide-react';
import axios from 'axios';
import { API_BASE_URL, SUPABASE_URL, SUPABASE_ANON_KEY } from '../../config';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

const DISEASES = [
    { id: 'alzheimers', name: "Alzheimer's" },
    { id: 'cancer', name: 'Cancer' },
    { id: 'diabetes', name: 'Diabetes' },
    { id: 'parkinson', name: "Parkinson's" },
    { id: 'cardiovascular', name: 'Cardiovascular' },
];

export default function BatchUploader() {
    const [file, setFile] = useState(null);
    const [disease, setDisease] = useState('alzheimers');
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile && selectedFile.name.endsWith('.csv')) {
            setFile(selectedFile);
            setError(null);
        } else {
            setError('Please upload a CSV file');
            setFile(null);
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const { data: { session } } = await supabase.auth.getSession();

            const formData = new FormData();
            formData.append('file', file);

            const response = await axios.post(
                `${API_BASE_URL}/qsar/predict/batch?disease_target=${disease}`,
                formData,
                {
                    headers: {
                        'Authorization': `Bearer ${session?.access_token}`,
                        'Content-Type': 'multipart/form-data'
                    }
                }
            );

            setResults(response.data);
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || 'Batch prediction failed');
        } finally {
            setLoading(false);
        }
    };

    const downloadResults = () => {
        if (!results?.predictions) return;

        const csv = [
            ['compound_name', 'smiles', 'prediction', 'score', 'confidence', 'interpretation'].join(','),
            ...results.predictions.map(p => [
                p.compound_name || '',
                `"${p.smiles}"`,
                p.prediction,
                p.score.toFixed(4),
                p.confidence.toFixed(4),
                `"${p.interpretation || ''}"`
            ].join(','))
        ].join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `batch_predictions_${disease}_${Date.now()}.csv`;
        a.click();
    };

    return (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                <FileSpreadsheet size={20} className="text-indigo-600" />
                Batch Prediction
            </h3>

            <p className="text-sm text-slate-500 mb-4">
                Upload a CSV file with a <code className="bg-slate-100 px-1 rounded">smiles</code> column to predict activity for multiple compounds at once.
            </p>

            {/* Disease Selector */}
            <div className="mb-4">
                <label className="block text-sm font-medium text-slate-700 mb-2">Disease Target</label>
                <select
                    value={disease}
                    onChange={(e) => setDisease(e.target.value)}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
                >
                    {DISEASES.map(d => (
                        <option key={d.id} value={d.id}>{d.name}</option>
                    ))}
                </select>
            </div>

            {/* File Upload */}
            <div
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${file ? 'border-green-300 bg-green-50' : 'border-slate-300 hover:border-indigo-300 hover:bg-indigo-50'
                    }`}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    className="hidden"
                />
                {file ? (
                    <div className="flex items-center justify-center gap-2 text-green-600">
                        <CheckCircle size={20} />
                        <span className="font-medium">{file.name}</span>
                    </div>
                ) : (
                    <div className="text-slate-500">
                        <Upload size={24} className="mx-auto mb-2" />
                        <span>Click to upload CSV or drag and drop</span>
                    </div>
                )}
            </div>

            {/* Error */}
            {error && (
                <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm flex items-center gap-2">
                    <AlertCircle size={16} />
                    {error}
                </div>
            )}

            {/* Upload Button */}
            <button
                onClick={handleUpload}
                disabled={!file || loading}
                className="mt-4 w-full py-3 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-700 transition-colors disabled:bg-slate-300 disabled:cursor-not-allowed"
            >
                {loading ? 'Processing...' : 'Run Batch Prediction'}
            </button>

            {/* Results */}
            {results && (
                <div className="mt-6">
                    <div className="flex items-center justify-between mb-4">
                        <h4 className="font-semibold text-slate-900">
                            Results ({results.total} compounds)
                        </h4>
                        <button
                            onClick={downloadResults}
                            className="flex items-center gap-1 text-sm text-indigo-600 hover:text-indigo-700"
                        >
                            <Download size={16} />
                            Download CSV
                        </button>
                    </div>

                    <div className="max-h-64 overflow-y-auto border border-slate-200 rounded-lg">
                        <table className="w-full text-sm">
                            <thead className="bg-slate-50 sticky top-0">
                                <tr>
                                    <th className="text-left p-2">SMILES</th>
                                    <th className="text-center p-2">Prediction</th>
                                    <th className="text-right p-2">Score</th>
                                </tr>
                            </thead>
                            <tbody>
                                {results.predictions.map((p, idx) => (
                                    <tr key={idx} className="border-t border-slate-100">
                                        <td className="p-2 font-mono text-xs truncate max-w-[200px]" title={p.smiles}>
                                            {p.smiles.slice(0, 30)}...
                                        </td>
                                        <td className="p-2 text-center">
                                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${p.prediction === 'Active' ? 'bg-green-100 text-green-700' :
                                                    p.prediction === 'Moderate' ? 'bg-yellow-100 text-yellow-700' :
                                                        'bg-red-100 text-red-700'
                                                }`}>
                                                {p.prediction}
                                            </span>
                                        </td>
                                        <td className="p-2 text-right font-medium">
                                            {(p.score * 100).toFixed(1)}%
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
