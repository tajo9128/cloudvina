import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { supabase } from '../supabaseClient';
import SEOHelmet from '../components/SEOHelmet';
import { Play, Upload, Settings, Activity, Clock, Zap, Database, Terminal, CheckCircle2, AlertCircle, Cpu } from 'lucide-react';

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
    const [logs, setLogs] = useState([]);

    // Add log entry
    const addLog = (message, type = 'info') => {
        setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), message, type }]);
    };

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            setPdbFile(file);
            const reader = new FileReader();
            reader.onload = (event) => {
                setPdbContent(event.target.result);
                addLog(`Loaded PDB file: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
            };
            reader.readAsText(file);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);
        setLogs([]); // Clear previous logs
        addLog('Initializing simulation request...', 'info');

        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session) {
                alert('Please login first');
                return;
            }

            addLog('Authenticating with cloud cluster...', 'info');

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

            if (!response.ok) throw new Error('Failed to submit job');

            const data = await response.json();
            setJobId(data.job_id);
            addLog(`Job submitted successfully [ID: ${data.job_id.slice(0, 8)}]`, 'success');
            addLog('Queued for GPU allocation...', 'info');

            pollJobStatus(data.job_id, session.access_token);

        } catch (error) {
            console.error('Error:', error);
            addLog(`Submission failed: ${error.message}`, 'error');
            alert('Error submitting job: ' + error.message);
        } finally {
            setIsSubmitting(false);
        }
    };

    const pollJobStatus = async (id, token) => {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`${import.meta.env.VITE_API_URL}/md/status/${id}`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                });

                if (response.ok) {
                    const status = await response.json();
                    setJobStatus(status);

                    // Update logs based on status
                    if (status.status === 'PROGRESS') {
                        // Only add log if distinct from last
                        // Simplified for demo: just showing status
                    }

                    if (status.status === 'SUCCESS') {
                        addLog('Simulation completed successfully!', 'success');
                        clearInterval(interval);
                    } else if (status.status === 'FAILURE') {
                        addLog(`Simulation failed: ${status.error}`, 'error');
                        clearInterval(interval);
                    }
                }
            } catch (error) {
                console.error('Error polling status:', error);
            }
        }, 5000);
    };

    const getProgressPercent = () => {
        if (!jobStatus) return 0;
        if (jobStatus.status === 'PROGRESS' && jobStatus.info?.progress) return jobStatus.info.progress;
        if (jobStatus.status === 'SUCCESS') return 100;
        return 0;
    };

    // Credit Cost Calculation (Example: 1ns = 10 credits)
    const durationNs = (config.steps * 0.002); // 2fs timestep
    const creditCost = Math.ceil(durationNs * 100);

    return (
        <div className="min-h-screen bg-slate-50 text-slate-900 font-sans pb-20">
            <SEOHelmet
                title="Molecular Dynamics Studio | BioDockify"
                description="High-performance cloud molecular dynamics. Run all-atom simulations with AMBER forcefields on GPU clusters."
                keywords="molecular dynamics, openmm, amber, protein simulation, cloud computing, gpu acceleration"
                canonical="https://biodockify.com/md-simulation"
            />

            {/* Header */}
            <div className="bg-slate-900 text-white pt-24 pb-32 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-32 bg-indigo-500/20 rounded-full blur-3xl transform translate-x-1/2 -translate-y-1/2"></div>
                <div className="container mx-auto px-6 relative z-10">
                    <h1 className="text-4xl md:text-5xl font-bold mb-4 tracking-tight">Molecular Dynamics Console</h1>
                    <p className="text-xl text-indigo-200 max-w-2xl">
                        Configure, deploy, and analyze all-atom simulations on our high-performance GPU cloud.
                    </p>
                </div>
            </div>

            <main className="container mx-auto px-6 -mt-20 relative z-20">
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

                    {/* LEFT COLUMN: Configuration Wizard */}
                    <div className="lg:col-span-7 space-y-6">
                        {!jobId ? (
                            <div className="bg-white rounded-3xl shadow-xl overflow-hidden border border-slate-200">
                                <div className="bg-slate-50 px-8 py-4 border-b border-slate-200 flex items-center justify-between">
                                    <div className="flex items-center gap-2 font-bold text-slate-700">
                                        <Settings className="w-5 h-5 text-indigo-600" />
                                        <span>Simulation Parameters</span>
                                    </div>
                                    <div className="text-xs font-mono text-slate-400">OpenMM Engine v8.0</div>
                                </div>
                                <form onSubmit={handleSubmit} className="p-8 space-y-8">

                                    {/* 1. Structure Upload */}
                                    <div className="space-y-4">
                                        <label className="text-sm font-bold text-slate-900 uppercase tracking-wide flex items-center gap-2">
                                            <span className="w-6 h-6 rounded-full bg-slate-900 text-white flex items-center justify-center text-xs">1</span>
                                            Input Structure
                                        </label>
                                        <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${pdbFile ? 'border-emerald-400 bg-emerald-50/50' : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50'}`}>
                                            <input
                                                type="file"
                                                accept=".pdb"
                                                onChange={handleFileUpload}
                                                className="hidden"
                                                id="pdb-upload"
                                            />
                                            <label htmlFor="pdb-upload" className="cursor-pointer block">
                                                {pdbFile ? (
                                                    <div className="flex flex-col items-center">
                                                        <div className="w-12 h-12 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center mb-3">
                                                            <CheckCircle2 size={24} />
                                                        </div>
                                                        <div className="font-bold text-emerald-900">{pdbFile.name}</div>
                                                        <div className="text-sm text-emerald-700">{(pdbFile.size / 1024).toFixed(1)} KB</div>
                                                    </div>
                                                ) : (
                                                    <div className="flex flex-col items-center">
                                                        <div className="w-12 h-12 bg-indigo-50 text-indigo-600 rounded-full flex items-center justify-center mb-3">
                                                            <Upload size={24} />
                                                        </div>
                                                        <div className="font-bold text-slate-700">Drop PDB file here</div>
                                                        <div className="text-sm text-slate-400">or click to browse</div>
                                                    </div>
                                                )}
                                            </label>
                                        </div>
                                    </div>

                                    {/* 2. System Configuration */}
                                    <div className="space-y-4">
                                        <label className="text-sm font-bold text-slate-900 uppercase tracking-wide flex items-center gap-2">
                                            <span className="w-6 h-6 rounded-full bg-slate-900 text-white flex items-center justify-center text-xs">2</span>
                                            System Builder
                                        </label>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                            <div className="space-y-2">
                                                <label className="text-xs font-bold text-slate-500 uppercase">Force Field</label>
                                                <select
                                                    value={config.forcefield}
                                                    onChange={(e) => setConfig({ ...config, forcefield: e.target.value })}
                                                    className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl font-medium focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                                                >
                                                    <option value="amber14-all.xml">AMBER14 (Recommended)</option>
                                                    <option value="charmm36.xml">CHARMM36</option>
                                                </select>
                                            </div>
                                            <div className="space-y-2">
                                                <label className="text-xs font-bold text-slate-500 uppercase">Temperature (K)</label>
                                                <input
                                                    type="number"
                                                    value={config.temperature}
                                                    onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
                                                    className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl font-medium focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                                                />
                                            </div>
                                        </div>
                                    </div>

                                    {/* 3. Production Run */}
                                    <div className="space-y-4">
                                        <label className="text-sm font-bold text-slate-900 uppercase tracking-wide flex items-center gap-2">
                                            <span className="w-6 h-6 rounded-full bg-slate-900 text-white flex items-center justify-center text-xs">3</span>
                                            Production Settings
                                        </label>
                                        <div>
                                            <div className="flex justify-between mb-2">
                                                <span className="text-sm font-medium text-slate-600">Simulation Duration</span>
                                                <span className="font-bold text-indigo-600">{durationNs.toFixed(3)} ns</span>
                                            </div>
                                            <input
                                                type="range"
                                                min="1000"
                                                max="50000"
                                                step="1000"
                                                value={config.steps}
                                                onChange={(e) => setConfig({ ...config, steps: parseInt(e.target.value) })}
                                                className="w-full"
                                            />
                                            <div className="flex justify-between text-xs text-slate-400 mt-1 font-mono">
                                                <span>2 ps (Test)</span>
                                                <span>100 ps (Production)</span>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Submit Action */}
                                    <div className="pt-6 border-t border-slate-100">
                                        <div className="flex items-center justify-between mb-4">
                                            <span className="text-sm text-slate-500 font-medium">Estimated Cost</span>
                                            <div className="flex items-center gap-1 text-slate-900 font-bold">
                                                <Zap className="w-4 h-4 text-amber-500 fill-amber-500" />
                                                <span>{creditCost} Credits</span>
                                            </div>
                                        </div>
                                        <button
                                            type="submit"
                                            disabled={isSubmitting || !pdbFile}
                                            className="w-full py-4 bg-slate-900 text-white font-bold rounded-xl shadow-lg hover:bg-slate-800 hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                                        >
                                            {isSubmitting ? <span className="animate-spin">⏳</span> : <Play size={20} fill="currentColor" />}
                                            {isSubmitting ? 'Initialize System...' : 'Deploy Simulation'}
                                        </button>
                                    </div>

                                </form>
                            </div>
                        ) : (
                            /* Job Monitoring View */
                            <div className="bg-white rounded-3xl shadow-xl overflow-hidden border border-slate-200 p-8 text-center">
                                <div className="w-20 h-20 bg-indigo-50 text-indigo-600 rounded-full flex items-center justify-center mx-auto mb-6">
                                    {jobStatus?.status === 'SUCCESS' ? <CheckCircle2 size={40} className="text-emerald-600" /> : <Activity size={40} className="animate-pulse" />}
                                </div>
                                <h2 className="text-2xl font-bold text-slate-900 mb-2">
                                    {jobStatus?.status === 'SUCCESS' ? 'Simulation Complete' : 'Simulation in Progress'}
                                </h2>
                                <p className="text-slate-500 mb-8 max-w-md mx-auto">
                                    {jobStatus?.status === 'SUCCESS' ? 'Trajectory analysis and energy calculations are ready for review.' : 'The high-performance cluster is currently processing your molecular dynamics job.'}
                                </p>

                                {jobStatus?.status === 'SUCCESS' ? (
                                    <Link to={`/md-results/${jobId}`} className="inline-flex items-center gap-2 px-8 py-4 bg-emerald-600 text-white font-bold rounded-xl hover:bg-emerald-700 transition-colors shadow-lg shadow-emerald-200">
                                        View Results Dashboard
                                        <Activity size={20} />
                                    </Link>
                                ) : (
                                    <div className="mb-8">
                                        <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                                            <div className="h-full bg-indigo-600 transition-all duration-500" style={{ width: `${getProgressPercent()}%` }}></div>
                                        </div>
                                        <div className="flex justify-between text-xs font-mono text-slate-500 mt-2">
                                            <span>Progress</span>
                                            <span>{getProgressPercent()}%</span>
                                        </div>
                                    </div>
                                )}

                                <button onClick={() => { setJobId(null); setJobStatus(null); setPdbFile(null); setLogs([]); }} className="block mx-auto mt-6 text-slate-400 hover:text-slate-600 text-sm font-medium">
                                    Start New Experiment
                                </button>
                            </div>
                        )}
                    </div>

                    {/* RIGHT COLUMN: Terminal & Info */}
                    <div className="lg:col-span-5 space-y-6">

                        {/* Live Terminal */}
                        <div className="bg-slate-900 rounded-2xl shadow-xl overflow-hidden border border-slate-700 font-mono text-sm">
                            <div className="bg-slate-800 px-4 py-3 border-b border-slate-700 flex items-center gap-2">
                                <Terminal className="w-4 h-4 text-emerald-400" />
                                <span className="text-slate-300 font-bold">System Log</span>
                            </div>
                            <div className="p-4 h-[400px] overflow-y-auto text-slate-300 space-y-2">
                                <div className="text-slate-500"># Waiting for job submission...</div>
                                {logs.map((log, i) => (
                                    <div key={i} className="flex gap-3">
                                        <span className="text-slate-600 shrink-0">[{log.time}]</span>
                                        <span className={log.type === 'error' ? 'text-red-400' : log.type === 'success' ? 'text-emerald-400' : 'text-slate-300'}>
                                            {log.type === 'info' && '> '}
                                            {log.message}
                                        </span>
                                    </div>
                                ))}
                                {isSubmitting && (
                                    <div className="animate-pulse text-indigo-400">_</div>
                                )}
                            </div>
                        </div>

                        {/* System Specs Card */}
                        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                            <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <Cpu className="w-5 h-5 text-indigo-500" />
                                Cluster Specifications
                            </h3>
                            <div className="space-y-3 text-sm">
                                <div className="flex justify-between pb-2 border-b border-slate-100">
                                    <span className="text-slate-500">Engine</span>
                                    <span className="font-bold text-slate-700">OpenMM 8.0 (CUDA)</span>
                                </div>
                                <div className="flex justify-between pb-2 border-b border-slate-100">
                                    <span className="text-slate-500">Hardware</span>
                                    <span className="font-bold text-slate-700">NVIDIA Tesla T4</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-slate-500">Precision</span>
                                    <span className="font-bold text-slate-700">Mixed (SPFP)</span>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </main>
        </div>
    );
};

export default MDSimulationPage;
