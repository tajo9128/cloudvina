import React, { useState, useEffect } from 'react';
import { supabase } from '../supabaseClient';

export default function MDStabilityPage() {
    const [activeTab, setActiveTab] = useState('quick'); // 'quick' or 'simulate'

    // Quick Analysis State
    const [rmsd, setRmsd] = useState('');
    const [rmsf, setRmsf] = useState('');
    const [molName, setMolName] = useState('');

    // Simulation State
    const [simFile, setSimFile] = useState(null);
    const [simMolName, setSimMolName] = useState('');
    const [simStatus, setSimStatus] = useState(null); // 'uploading', 'submitting', 'success', 'error'

    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [history, setHistory] = useState([]);
    const [error, setError] = useState(null);

    // Fetch History on Mount
    useEffect(() => {
        fetchHistory();
    }, []);

    const fetchHistory = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const token = session?.access_token;
            const res = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/md-analysis/history`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                const data = await res.json();
                setHistory(data);
            }
        } catch (err) {
            console.error("Failed to load history", err);
        }
    };

    const handleQuickAnalyze = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const { data: { session } } = await supabase.auth.getSession();
            const token = session?.access_token;

            const payload = {
                molecule_name: molName,
                rmsd: parseFloat(rmsd),
                rmsf: parseFloat(rmsf)
            };

            const res = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/md-analysis/calculate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Analysis Failed');

            setResult(data);
            fetchHistory();
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleSimulationSubmit = async (e) => {
        e.preventDefault();
        if (!simFile) {
            setError("Please select a PDB file.");
            return;
        }

        setLoading(true);
        setSimStatus('uploading');
        setError(null);

        try {
            const { data: { session } } = await supabase.auth.getSession();
            const token = session?.access_token;

            // 1. Get Presigned URL
            const uploadRes = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/md-analysis/upload-pdb`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                body: JSON.stringify({ filename: simFile.name })
            });

            if (!uploadRes.ok) throw new Error("Failed to get upload URL");
            const { upload_url, s3_key } = await uploadRes.json();

            // 2. Upload to S3
            const s3Res = await fetch(upload_url, {
                method: 'PUT',
                body: simFile,
                headers: { 'Content-Type': 'chemical/x-pdb' }
            });
            if (!s3Res.ok) throw new Error("Failed to upload PDB to S3");

            // 3. Trigger Simulation
            setSimStatus('submitting');
            const triggerRes = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/md-analysis/start-simulation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                body: JSON.stringify({
                    molecule_name: simMolName,
                    pdb_s3_key: s3_key
                })
            });

            if (!triggerRes.ok) throw new Error("Failed to start simulation");
            const triggerData = await triggerRes.json();

            setSimStatus('success');
            setResult({
                special_message: `Simulation Started! Job ID: ${triggerData.job_id}`,
                status: 'SUBMITTED'
            });
            fetchHistory();

        } catch (err) {
            console.error(err);
            setError(err.message);
            setSimStatus('error');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold mb-2 text-gray-800">🧬 MD Stability Engine</h1>
            <p className="mb-6 text-gray-600">
                Rigorous Physics-Based Simulation (OpenMM) + AI Stability Scoring (Ensemble).
            </p>

            {/* Tabs */}
            <div className="flex border-b border-gray-200 mb-6">
                <button
                    onClick={() => setActiveTab('quick')}
                    className={`py-2 px-4 font-medium ${activeTab === 'quick' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
                >
                    ⚡ Quick Score (Enter Metrics)
                </button>
                <button
                    onClick={() => setActiveTab('simulate')}
                    className={`py-2 px-4 font-medium ${activeTab === 'simulate' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
                >
                    🧪 Full Simulation (Upload PDB)
                </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Input Area */}
                <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">

                    {activeTab === 'quick' ? (
                        <form onSubmit={handleQuickAnalyze} className="space-y-4">
                            <h2 className="text-xl font-semibold mb-4">Manual Input</h2>
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Molecule Name</label>
                                <input type="text" required value={molName} onChange={(e) => setMolName(e.target.value)}
                                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2 border" placeholder="e.g. Donepezil-Complex" />
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700">RMSD (Å)</label>
                                    <input type="number" step="0.01" required value={rmsd} onChange={(e) => setRmsd(e.target.value)}
                                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2 border" placeholder="1.5" />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700">RMSF (Å)</label>
                                    <input type="number" step="0.01" required value={rmsf} onChange={(e) => setRmsf(e.target.value)}
                                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2 border" placeholder="0.8" />
                                </div>
                            </div>
                            <button type="submit" disabled={loading} className="w-full py-2 px-4 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:bg-gray-400">
                                {loading ? 'Calculating...' : 'Get AI Score'}
                            </button>
                        </form>
                    ) : (
                        <form onSubmit={handleSimulationSubmit} className="space-y-4">
                            <h2 className="text-xl font-semibold mb-4">Run Simulation on AWS</h2>
                            <div className="bg-blue-50 p-4 rounded text-sm text-blue-800 mb-4">
                                This will run a <b>Full OpenMM Simulation</b> on AWS Batch, then calculate RMSD/RMSF, and finally produce an AI Score.
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Molecule Name</label>
                                <input type="text" required value={simMolName} onChange={(e) => setSimMolName(e.target.value)}
                                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2 border" placeholder="e.g. New-Drug-Candidates" />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Input PDB File</label>
                                <input type="file" required accept=".pdb" onChange={(e) => setSimFile(e.target.files[0])}
                                    className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100" />
                            </div>
                            <button type="submit" disabled={loading} className="w-full py-2 px-4 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:bg-gray-400">
                                {loading ? (simStatus === 'uploading' ? 'Uploading...' : 'Submitting to Batch...') : 'Start Simulation Job'}
                            </button>
                        </form>
                    )}

                    {error && <div className="mt-4 text-red-600 bg-red-50 p-3 rounded">{error}</div>}
                </div>

                {/* Results Area */}
                <div>
                    {result ? (
                        <div className={`p-6 rounded-lg shadow-lg border-2 ${result.status === 'SUBMITTED' ? 'bg-blue-50 border-blue-200' : (result.score > 75 ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200')}`}>
                            <h2 className="text-xl font-semibold mb-2">
                                {result.status === 'SUBMITTED' ? 'Job Submitted' : 'Analysis Result'}
                            </h2>

                            {result.status === 'SUBMITTED' ? (
                                <div className="text-blue-800">
                                    <p className="text-lg">🚀 Simulation Queued on AWS Batch.</p>
                                    <p className="text-sm mt-2">{result.special_message}</p>
                                    <p className="text-xs mt-4 text-gray-500">Check the History table below for status updates (SUCCESS/FAILED).</p>
                                </div>
                            ) : (
                                <div className="flex items-center justify-between mt-4">
                                    <div>
                                        <p className="text-sm text-gray-500">Stability Score</p>
                                        <p className={`text-5xl font-bold ${result.score > 75 ? 'text-green-600' : 'text-yellow-600'}`}>
                                            {result.score?.toFixed(1)} / 100
                                        </p>
                                    </div>
                                    <div className="text-right">
                                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${result.score > 75 ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
                                            {result.score > 75 ? "High Stability" : "Moderate Stability"}
                                        </span>
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="bg-gray-50 p-6 rounded-lg border border-dashed border-gray-300 h-full flex items-center justify-center text-gray-400">
                            {loading ? 'Processing...' : 'Results / Status will appear here...'}
                        </div>
                    )}
                </div>
            </div>

            {/* History Table */}
            <div className="mt-12">
                <h3 className="text-lg font-semibold mb-4">Previous Jobs</h3>
                <div className="overflow-x-auto bg-white shadow rounded-lg">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Molecule</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">RMSD / RMSF</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                            {history.map((job) => (
                                <tr key={job.id}>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{job.molecule_name}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                                            ${job.status === 'SUCCESS' ? 'bg-green-100 text-green-800' :
                                                job.status === 'SUBMITTED' ? 'bg-blue-100 text-blue-800' :
                                                    'bg-gray-100 text-gray-800'}`}>
                                            {job.status}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-700">
                                        {job.md_score ? job.md_score.toFixed(1) : '-'}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                        {job.rmsd}/{job.rmsf}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{new Date(job.created_at).toLocaleDateString()}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
