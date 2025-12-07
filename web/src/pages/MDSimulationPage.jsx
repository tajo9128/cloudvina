import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { supabase } from '../supabaseClient';
import SEOHelmet from '../components/SEOHelmet';

const MDSimulationPage = () => {
    const [pdbFile, setPdbFile] = useState(null);
    const [pdbContent, setPdbContent] = useState('');
    const [config, setConfig] = useState({
        temperature: 300,
        steps: 5000,
        forcefield: 'amber14-all.xml',
        water: 'amber14/tip3pfb.xml'
    });
    const [jobId, setJobId] = useState(null);
    const [jobStatus, setJobStatus] = useState(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            setPdbFile(file);
            const reader = new FileReader();
            reader.onload = (event) => {
                setPdbContent(event.target.result);
            };
            reader.readAsText(file);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);

        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session) {
                alert('Please login first');
                return;
            }

            const response = await fetch(`${import.meta.env.VITE_API_URL}/md/submit`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session.access_token}`
                },
                body: JSON.stringify({
                    pdb_content: pdbContent,
                    config: config
                })
            });

            if (!response.ok) {
                throw new Error('Failed to submit job');
            }

            const data = await response.json();
            setJobId(data.job_id);

            // Start polling for status
            pollJobStatus(data.job_id, session.access_token);

        } catch (error) {
            console.error('Error:', error);
            alert('Error submitting job: ' + error.message);
        } finally {
            setIsSubmitting(false);
        }
    };

    const pollJobStatus = async (id, token) => {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(
                    `${import.meta.env.VITE_API_URL}/md/status/${id}`,
                    {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    }
                );

                if (response.ok) {
                    const status = await response.json();
                    setJobStatus(status);

                    if (status.status === 'SUCCESS' || status.status === 'FAILURE') {
                        clearInterval(interval);
                    }
                }
            } catch (error) {
                console.error('Error polling status:', error);
            }
        }, 5000); // Poll every 5 seconds
    };

    const getProgressPercent = () => {
        if (!jobStatus) return 0;
        if (jobStatus.status === 'PROGRESS' && jobStatus.info?.progress) {
            return jobStatus.info.progress;
        }
        if (jobStatus.status === 'SUCCESS') return 100;
        return 0;
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 py-12 px-4">
            <SEOHelmet
                title="Free Molecular Dynamics Simulation | OpenMM Cloud - BioDockify"
                description="Run all-atom molecular dynamics simulations online using OpenMM. Analyze protein stability, RMSD, and RMSF with free GPU acceleration."
                keywords="molecular dynamics online, free md simulation, openmm cloud, protein stability analysis, rmsd calculation online"
                canonical="https://biodockify.com/md-simulation"
            />
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="text-center mb-10">
                    <h1 className="text-4xl font-bold text-gray-900 mb-3">
                        Molecular Dynamics Simulation
                    </h1>
                    <p className="text-lg text-gray-600">
                        Powered by OpenMM & Google Colab (Free GPU)
                    </p>
                </div>

                {!jobId ? (
                    /* Submission Form */
                    <div className="bg-white rounded-2xl shadow-xl p-8">
                        <form onSubmit={handleSubmit} className="space-y-6">
                            {/* PDB Upload */}
                            <div>
                                <label className="block text-sm font-semibold text-gray-700 mb-2">
                                    Upload Protein Structure (PDB)
                                </label>
                                <input
                                    type="file"
                                    accept=".pdb"
                                    onChange={handleFileUpload}
                                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                    required
                                />
                                {pdbFile && (
                                    <p className="text-sm text-green-600 mt-2">
                                        ✓ Loaded: {pdbFile.name}
                                    </p>
                                )}
                            </div>

                            {/* Temperature */}
                            <div>
                                <label className="block text-sm font-semibold text-gray-700 mb-2">
                                    Temperature (K)
                                </label>
                                <input
                                    type="number"
                                    value={config.temperature}
                                    onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
                                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                                    min="0"
                                    step="1"
                                />
                            </div>

                            {/* Simulation Steps */}
                            <div>
                                <label className="block text-sm font-semibold text-gray-700 mb-2">
                                    Simulation Steps (10ps = 5000 steps)
                                </label>
                                <input
                                    type="number"
                                    value={config.steps}
                                    onChange={(e) => setConfig({ ...config, steps: parseInt(e.target.value) })}
                                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                                    min="1000"
                                    step="1000"
                                />
                                <p className="text-sm text-gray-500 mt-1">
                                    Duration: ~{(config.steps * 0.002 / 1000).toFixed(2)} ns
                                </p>
                            </div>

                            {/* Forcefield Selection */}
                            <div>
                                <label className="block text-sm font-semibold text-gray-700 mb-2">
                                    Force Field
                                </label>
                                <select
                                    value={config.forcefield}
                                    onChange={(e) => setConfig({ ...config, forcefield: e.target.value })}
                                    className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                                >
                                    <option value="amber14-all.xml">AMBER14</option>
                                    <option value="charmm36.xml">CHARMM36</option>
                                </select>
                            </div>

                            {/* Submit Button */}
                            <button
                                type="submit"
                                disabled={isSubmitting || !pdbFile}
                                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white font-bold py-4 px-6 rounded-lg hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg"
                            >
                                {isSubmitting ? 'Submitting...' : 'Start Simulation'}
                            </button>
                        </form>

                        {/* Info Box */}
                        <div className="mt-6 bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                            <p className="text-sm text-blue-800">
                                <strong>💡 How it works:</strong> Your job is queued in our Redis system.
                                A worker running on Google Colab (with free GPU!) will pick it up and run the simulation.
                                Keep this page open to monitor progress.
                            </p>
                        </div>
                    </div>
                ) : (
                    /* Job Status */
                    <div className="bg-white rounded-2xl shadow-xl p-8">
                        <h2 className="text-2xl font-bold text-gray-900 mb-6">
                            Job Status
                        </h2>

                        {/* Job ID */}
                        <div className="mb-6">
                            <p className="text-sm text-gray-600">Job ID</p>
                            <p className="font-mono text-sm bg-gray-100 px-3 py-2 rounded">
                                {jobId}
                            </p>
                        </div>

                        {/* Progress Bar */}
                        <div className="mb-6">
                            <div className="flex justify-between text-sm text-gray-600 mb-2">
                                <span>Progress</span>
                                <span>{getProgressPercent()}%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                                <div
                                    className="bg-gradient-to-r from-purple-600 to-blue-600 h-full transition-all duration-500"
                                    style={{ width: `${getProgressPercent()}%` }}
                                />
                            </div>
                        </div>

                        {/* Status Message */}
                        <div className="mb-6">
                            <p className="text-sm text-gray-600 mb-2">Status</p>
                            <div className="bg-gray-50 px-4 py-3 rounded-lg">
                                {jobStatus?.status === 'PENDING' && (
                                    <p className="text-yellow-600">⏳ Queued - Waiting for worker...</p>
                                )}
                                {jobStatus?.status === 'PROGRESS' && (
                                    <p className="text-blue-600">
                                        🔄 {jobStatus.info?.status || 'Running...'}
                                    </p>
                                )}
                                {jobStatus?.status === 'SUCCESS' && (
                                    <p className="text-green-600">✅ Simulation complete!</p>
                                )}
                                {jobStatus?.status === 'FAILURE' && (
                                    <p className="text-red-600">❌ Simulation failed</p>
                                )}
                            </div>
                        </div>

                        {/* Results (if completed) */}
                        {jobStatus?.status === 'SUCCESS' && jobStatus.result && (
                            <div className="mt-6 bg-green-50 border-l-4 border-green-500 p-4 rounded">
                                <h3 className="font-bold text-green-900 mb-2">Simulation Complete!</h3>
                                <p className="text-sm text-green-800 mb-3">
                                    Your MD simulation finished successfully. View detailed analysis and trajectory.
                                </p>
                                <Link
                                    to={`/md-results/${jobId}`}
                                    className="inline-block px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold transition-colors"
                                >
                                    📊 View Results & Analysis
                                </Link>
                            </div>
                        )}

                        {/* Error (if failed) */}
                        {jobStatus?.status === 'FAILURE' && jobStatus.error && (
                            <div className="mt-6 bg-red-50 border-l-4 border-red-500 p-4 rounded">
                                <h3 className="font-bold text-red-900 mb-2">Error</h3>
                                <p className="text-sm text-red-800">{jobStatus.error}</p>
                            </div>
                        )}

                        {/* New Job Button */}
                        <button
                            onClick={() => {
                                setJobId(null);
                                setJobStatus(null);
                                setPdbFile(null);
                                setPdbContent('');
                            }}
                            className="mt-6 w-full bg-gray-200 text-gray-800 font-bold py-3 px-6 rounded-lg hover:bg-gray-300 transition-colors"
                        >
                            Start New Simulation
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default MDSimulationPage;
