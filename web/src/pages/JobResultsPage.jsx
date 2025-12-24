import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import ExportButtons from '../components/ExportButtons'
import DockingResultsTable from '../components/DockingResultsTable'
import InteractionTable from '../components/InteractionTable'
import DrugPropertiesPanel from '../components/DrugPropertiesPanel'
import { API_URL } from '../config'
import MoleculeViewer from '../components/MoleculeViewer'
import { ArrowRight, Download, FileText, Activity, Clock, Database, Layers, Zap, MoreHorizontal, FlaskConical, Play } from 'lucide-react'

export default function JobResultsPage() {
    const { jobId } = useParams()
    const [job, setJob] = useState(null)
    const [analysis, setAnalysis] = useState(null)
    const [interactions, setInteractions] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [elapsedTime, setElapsedTime] = useState(0)
    const [pdbqtData, setPdbqtData] = useState(null)
    const [receptorData, setReceptorData] = useState(null)

    // Multiple Pocket Results State
    const [detectedPockets, setDetectedPockets] = useState([])
    const [selectedPocketId, setSelectedPocketId] = useState(1)

    // Consensus Results State
    const [consensusResults, setConsensusResults] = useState(null)

    const ESTIMATED_DURATION = 300 // 5 minutes

    useEffect(() => {
        let timer
        if (job && ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(job.status)) {
            const startTime = job.created_at ? new Date(job.created_at).getTime() : Date.now()
            timer = setInterval(() => {
                const now = Date.now()
                setElapsedTime(Math.floor((now - startTime) / 1000))
            }, 1000)
        }
        return () => clearInterval(timer)
    }, [job])

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}m ${secs}s`
    }

    useEffect(() => {
        fetchJob()
        const interval = setInterval(() => {
            if (job && ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(job.status)) fetchJob()
        }, 5000)
        return () => clearInterval(interval)
    }, [jobId, job?.status])

    const fetchJob = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) throw new Error('Not authenticated')

            const res = await fetch(`${API_URL}/jobs/${jobId}`, {
                headers: { 'Authorization': `Bearer ${session.access_token}` }
            })

            if (!res.ok) throw new Error('Failed to fetch job')
            const data = await res.json()
            setJob(data)

            if (data.status === 'SUCCEEDED') {
                if (!analysis) fetchAnalysis(session.access_token)
                if (!interactions) fetchInteractions(session.access_token)
                if (detectedPockets.length === 0) fetchPockets(session.access_token)
                if (!pdbqtData) fetchStructure(data)
                if (data.download_urls?.results_json && !consensusResults) fetchConsensus(data.download_urls.results_json)
            }
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const fetchAnalysis = async (token) => {
        try {
            const res = await fetch(`${API_URL}/jobs/${jobId}/analysis`, { headers: { 'Authorization': `Bearer ${token}` } })
            if (res.ok) {
                const data = await res.json()
                setAnalysis(data.analysis)
            } else setAnalysis({ error: true })
        } catch (err) { setAnalysis({ error: true }) }
    }

    const fetchInteractions = async (token) => {
        try {
            const res = await fetch(`${API_URL}/jobs/${jobId}/interactions`, { headers: { 'Authorization': `Bearer ${token}` } })
            if (res.ok) {
                const data = await res.json()
                setInteractions(data.interactions)
            } else setInteractions({ error: true })
        } catch (err) { setInteractions({ error: true }) }
    }

    const fetchPockets = async (token) => {
        try {
            const res = await fetch(`${API_URL}/jobs/${jobId}/detect-cavities`, { method: 'POST', headers: { 'Authorization': `Bearer ${token}` } })
            if (res.ok) {
                const data = await res.json()
                if (data.cavities?.length > 0) setDetectedPockets(data.cavities)
            }
        } catch (err) { console.error(err) }
    }

    const fetchStructure = async (data) => {
        try {
            const pdbqtUrl = data.download_urls?.output_vina || data.download_urls?.output
            if (pdbqtUrl) {
                const res = await fetch(pdbqtUrl)
                setPdbqtData(await res.text())
            }
            if (data.download_urls?.receptor && !receptorData) {
                const res = await fetch(data.download_urls.receptor)
                if (res.ok) setReceptorData(await res.text())
            }
        } catch (err) { console.error(err) }
    }

    const fetchConsensus = async (url) => {
        try {
            const res = await fetch(url)
            if (res.ok) setConsensusResults(await res.json())
        } catch (err) { setConsensusResults({ error: "Failed to load consensus data" }) }
    }

    const [loadingMD, setLoadingMD] = useState(false)
    const handleRunMD = async () => {
        if (!confirm("Start Molecular Dynamics stability simulation? (Cost: 10 Credits)")) return;
        setLoadingMD(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!pdbqtData) throw new Error("Structure data not loaded yet.");
            const res = await fetch(`${API_URL}/md/submit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${session.access_token}` },
                body: JSON.stringify({ pdb_content: pdbqtData, config: { steps: 5000 } })
            })
            if (!res.ok) throw new Error("Failed to start MD");
            const data = await res.json()
            alert(`MD Simulation Started! Job ID: ${data.job_id}`);
        } catch (err) { alert("MD Start Failed: " + err.message); } finally { setLoadingMD(false); }
    }

    if (loading) return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center">
            <div className="w-16 h-16 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin"></div>
        </div>
    )

    if (error || !job) return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center">
            <div className="text-center">
                <h3 className="text-xl font-bold text-slate-800">Job Not Found</h3>
                <Link to="/dashboard" className="text-indigo-600 hover:underline mt-2 inline-block">Return to Dashboard</Link>
            </div>
        </div>
    )

    const progressPercentage = Math.min((elapsedTime / ESTIMATED_DURATION) * 100, 99)

    return (
        <div className="min-h-screen bg-slate-50 font-sans">
            {/* Header */}
            <div className="h-20 bg-white border-b border-slate-200 flex items-center justify-between px-8 sticky top-0 z-40 shadow-sm/50 backdrop-blur-md bg-white/90">
                <div className="flex items-center gap-6">
                    <Link to="/dashboard" className="w-10 h-10 flex items-center justify-center hover:bg-slate-100 rounded-full text-slate-400 transition-colors">
                        <ArrowRight className="rotate-180 w-5 h-5" />
                    </Link>
                    <div>
                        <div className="flex items-center gap-3">
                            <h1 className="text-xl font-bold text-slate-900 tracking-tight">Job Result Page</h1>
                            <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider ${job.status === 'SUCCEEDED' ? 'bg-emerald-100 text-emerald-700' :
                                job.status === 'FAILED' ? 'bg-red-100 text-red-700' :
                                    'bg-indigo-100 text-indigo-700 animate-pulse'
                                }`}>
                                {job.status}
                            </span>
                        </div>
                        <div className="text-xs text-slate-500 mt-1 font-mono">
                            ID: {jobId}
                        </div>
                    </div>
                </div>
                <div className="hidden md:flex">
                    <ExportButtons jobId={jobId} />
                </div>
            </div>

            <main className="max-w-[1600px] mx-auto p-6 lg:p-8">

                {/* Progress Banner */}
                {['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(job.status) && (
                    <div className="bg-indigo-900 rounded-2xl p-8 mb-8 text-white relative overflow-hidden shadow-xl">
                        <div className="absolute top-0 right-0 p-32 bg-indigo-500/20 rounded-full blur-3xl transform translate-x-1/2 -translate-y-1/2"></div>
                        <div className="relative z-10">
                            <div className="flex justify-between items-end mb-4">
                                <div>
                                    <h2 className="text-2xl font-bold mb-2 flex items-center gap-3">
                                        <Clock className="animate-spin-slow w-6 h-6 text-indigo-300" />
                                        Simulation Running
                                    </h2>
                                    <p className="text-indigo-200">Processing docking algorithm on high-performance cluster.</p>
                                </div>
                                <div className="text-4xl font-bold font-mono text-white opacity-20">{Math.round(progressPercentage)}%</div>
                            </div>
                            <div className="h-2 bg-indigo-950/50 rounded-full overflow-hidden">
                                <div className="h-full bg-gradient-to-r from-indigo-400 to-emerald-400 transition-all duration-1000 ease-out" style={{ width: `${progressPercentage}%` }} />
                            </div>
                        </div>
                    </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

                    {/* LEFT COLUMN (Data) - Span 7 */}
                    <div className="lg:col-span-7 space-y-8">

                        {/* 1. Key Metrics Card */}
                        <div className="bg-white rounded-3xl shadow-sm border border-slate-200 p-8">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                {/* Metadata */}
                                <div className="space-y-4">
                                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Experiment Details</h3>
                                    <div className="flex items-center gap-4 p-3 bg-slate-50 rounded-xl border border-slate-100">
                                        <div className="p-2 bg-indigo-100 text-indigo-600 rounded-lg"><Database size={18} /></div>
                                        <div className="overflow-hidden">
                                            <div className="text-xs text-slate-500 font-bold">Receptor</div>
                                            <div className="font-medium text-slate-900 truncate" title={job.receptor_filename}>{job.receptor_filename}</div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-4 p-3 bg-slate-50 rounded-xl border border-slate-100">
                                        <div className="p-2 bg-emerald-100 text-emerald-600 rounded-lg"><FlaskConical size={18} /></div>
                                        <div className="overflow-hidden">
                                            <div className="text-xs text-slate-500 font-bold">Ligand</div>
                                            <div className="font-medium text-slate-900 truncate" title={job.ligand_filename}>{job.ligand_filename}</div>
                                        </div>
                                    </div>
                                </div>

                                {/* Affinity Score */}
                                <div className="flex flex-col justify-center items-center text-center p-6 bg-gradient-to-br from-indigo-50 to-white border border-indigo-100 rounded-2xl">
                                    <div className="text-sm font-bold text-indigo-400 uppercase tracking-widest mb-2">Binding Affinity</div>
                                    <div className="text-5xl font-bold text-slate-900 tracking-tighter mb-1">
                                        {consensusResults?.best_affinity?.toFixed(1) || analysis?.best_affinity?.toFixed(1) || '-'}
                                    </div>
                                    <div className="text-lg font-medium text-slate-400">kcal/mol</div>
                                    {consensusResults && (
                                        <div className="flex gap-3 mt-4">
                                            <span className="text-xs px-2 py-1 bg-white border border-slate-200 rounded text-slate-500">Vina: {consensusResults.engines?.vina?.best_affinity?.toFixed(1) || '-'}</span>
                                            <span className="text-xs px-2 py-1 bg-white border border-slate-200 rounded text-slate-500">Gnina: {consensusResults.engines?.gnina?.best_affinity?.toFixed(1) || '-'}</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* 2. Downloads Vault */}
                        {job.status === 'SUCCEEDED' && job.download_urls && (
                            <div className="bg-slate-900 rounded-3xl shadow-xl overflow-hidden text-white relative">
                                <div className="p-8 relative z-10">
                                    <h3 className="flex items-center gap-2 font-bold text-lg mb-6">
                                        <Download className="text-indigo-400" size={20} /> Data Export Vault
                                    </h3>
                                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">

                                        {/* 0. Full Report (NEW) */}
                                        <a href={`${API_URL}/jobs/${jobId}/export/pdf`} download className="group col-span-1 lg:col-span-3 flex items-center justify-between p-4 bg-gradient-to-r from-red-600/20 to-orange-600/20 hover:from-red-600/30 hover:to-orange-600/30 border border-red-500/30 rounded-xl transition-all">
                                            <div className="flex items-center gap-3">
                                                <div className="p-2 bg-red-500 rounded-lg text-white shadow-lg shadow-red-500/30"><FileText size={20} /></div>
                                                <div>
                                                    <div className="font-bold text-sm text-red-100">Full Publication Report</div>
                                                    <div className="text-xs text-red-200/50">PDF (Methods, Results, Plots)</div>
                                                </div>
                                            </div>
                                            <Download size={16} className="text-red-400 group-hover:text-white transition-colors" />
                                        </a>

                                        {/* 0.5. PyMOL Session (NEW) */}
                                        <a href={`${API_URL}/jobs/${jobId}/export/pymol`} download className="group flex items-center justify-between p-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl transition-all">
                                            <div className="flex items-center gap-3">
                                                <div className="p-2 bg-orange-500/20 rounded-lg text-orange-400 group-hover:text-white group-hover:bg-orange-500 transition-colors"><Layers size={20} /></div>
                                                <div>
                                                    <div className="font-bold text-sm">PyMOL Session</div>
                                                    <div className="text-xs text-orange-200/50">Visualization Script (.pml)</div>
                                                </div>
                                            </div>
                                            <Download size={16} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                                        </a>

                                        {/* 1. Receptor */}
                                        {job.download_urls.receptor && (
                                            <a href={job.download_urls.receptor} className="group flex items-center justify-between p-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl transition-all">
                                                <div className="flex items-center gap-3">
                                                    <div className="p-2 bg-slate-700 rounded-lg text-slate-300 group-hover:text-white transition-colors"><Database size={20} /></div>
                                                    <div>
                                                        <div className="font-bold text-sm">Target Receptor</div>
                                                        <div className="text-xs text-slate-400">PDB/PDBQT</div>
                                                    </div>
                                                </div>
                                                <Download size={16} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                                            </a>
                                        )}

                                        {/* 2. Vina Output */}
                                        {(job.download_urls.output_vina || job.download_urls.output) && (
                                            <a href={job.download_urls.output_vina || job.download_urls.output} className="group flex items-center justify-between p-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl transition-all">
                                                <div className="flex items-center gap-3">
                                                    <div className="p-2 bg-blue-500/20 rounded-lg text-blue-400 group-hover:text-white group-hover:bg-blue-500 transition-colors"><Database size={20} /></div>
                                                    <div>
                                                        <div className="font-bold text-sm">Vina Structure</div>
                                                        <div className="text-xs text-blue-200/50">Output PDBQT</div>
                                                    </div>
                                                </div>
                                                <Download size={16} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                                            </a>
                                        )}

                                        {/* 3. Gnina Output */}
                                        {job.download_urls.output_gnina && (
                                            <a href={job.download_urls.output_gnina} className="group flex items-center justify-between p-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl transition-all">
                                                <div className="flex items-center gap-3">
                                                    <div className="p-2 bg-purple-500/20 rounded-lg text-purple-400 group-hover:text-white group-hover:bg-purple-500 transition-colors"><Activity size={20} /></div>
                                                    <div>
                                                        <div className="font-bold text-sm">Gnina Structure</div>
                                                        <div className="text-xs text-purple-200/50">CNN Output</div>
                                                    </div>
                                                </div>
                                                <Download size={16} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                                            </a>
                                        )}

                                        {/* 4. Results JSON */}
                                        {job.download_urls.results_json && (
                                            <a href={job.download_urls.results_json} className="group flex items-center justify-between p-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl transition-all">
                                                <div className="flex items-center gap-3">
                                                    <div className="p-2 bg-emerald-500/20 rounded-lg text-emerald-400 group-hover:text-white group-hover:bg-emerald-500 transition-colors"><FileText size={20} /></div>
                                                    <div>
                                                        <div className="font-bold text-sm">Consensus Metrics</div>
                                                        <div className="text-xs text-emerald-200/50">JSON Data</div>
                                                    </div>
                                                </div>
                                                <Download size={16} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                                            </a>
                                        )}

                                        {/* 5. Config */}
                                        {job.download_urls.config && (
                                            <a href={job.download_urls.config} className="group flex items-center justify-between p-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl transition-all">
                                                <div className="flex items-center gap-3">
                                                    <div className="p-2 bg-slate-500/20 rounded-lg text-slate-400 group-hover:text-white group-hover:bg-slate-500 transition-colors"><FileText size={20} /></div>
                                                    <div>
                                                        <div className="font-bold text-sm">Configuration</div>
                                                        <div className="text-xs text-slate-200/50">Parameters</div>
                                                    </div>
                                                </div>
                                                <Download size={16} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                                            </a>
                                        )}

                                        {/* 6. Logs */}
                                        {job.download_urls.log && (
                                            <a href={job.download_urls.log} className="group flex items-center justify-between p-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl transition-all">
                                                <div className="flex items-center gap-3">
                                                    <div className="p-2 bg-slate-500/20 rounded-lg text-slate-400 group-hover:text-white group-hover:bg-slate-500 transition-colors"><Activity size={20} /></div>
                                                    <div>
                                                        <div className="font-bold text-sm">Execution Log</div>
                                                        <div className="text-xs text-slate-200/50">Console Output</div>
                                                    </div>
                                                </div>
                                                <Download size={16} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                                            </a>
                                        )}
                                    </div>
                                </div>
                                <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-600/20 blur-[80px] rounded-full pointer-events-none -mr-16 -mt-16"></div>
                            </div>
                        )}

                        {/* 3. MD Simulation Call to Action */}
                        {job.status === 'SUCCEEDED' && (
                            <div className="bg-gradient-to-r from-violet-600 to-indigo-600 rounded-2xl p-1 shadow-lg">
                                <div className="bg-white rounded-xl p-6 flex flex-col md:flex-row items-center justify-between gap-6">
                                    <div className="flex items-center gap-4">
                                        <div className="p-3 bg-violet-50 text-violet-600 rounded-xl">
                                            <Activity size={24} />
                                        </div>
                                        <div>
                                            <h3 className="font-bold text-slate-900 text-lg">Validate with MD Simulation</h3>
                                            <p className="text-slate-500 text-sm">Run 10ns stability check via OpenMM (Cost: 10 Credits)</p>
                                        </div>
                                    </div>
                                    <button
                                        onClick={handleRunMD}
                                        disabled={loadingMD}
                                        className="px-6 py-3 bg-slate-900 text-white font-bold rounded-xl hover:bg-slate-800 transition-all shadow-xl hover:shadow-2xl flex items-center gap-2 disabled:opacity-50"
                                    >
                                        {loadingMD ? <span className="animate-spin">⌛</span> : <Play size={16} fill="white" />}
                                        <span>Run Simulation</span>
                                    </button>
                                </div>
                            </div>
                        )}

                        {/* 4. Consensus & Pockets (Tabs Style) */}
                        {[consensusResults, detectedPockets.length > 0].some(Boolean) && (
                            <div className="space-y-6">
                                {consensusResults && !consensusResults.error && (
                                    <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
                                        <h3 className="font-bold text-slate-900 flex items-center gap-2 mb-4"><Layers className="text-indigo-500" /> Consensus Breakdown</h3>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                            <div className="p-4 rounded-2xl bg-blue-50/50 border border-blue-100">
                                                <div className="text-xs font-bold text-blue-400 uppercase mb-1">Physics-Based</div>
                                                <div className="font-bold text-blue-900 text-lg">AutoDock Vina</div>
                                                <div className="text-2xl font-bold text-blue-600 mt-2">{consensusResults.engines?.vina?.best_affinity?.toFixed(1) || '-'} <span className="text-sm font-normal opacity-50">kcal/mol</span></div>
                                            </div>
                                            <div className="p-4 rounded-2xl bg-purple-50/50 border border-purple-100">
                                                <div className="text-xs font-bold text-purple-400 uppercase mb-1">Deep Learning</div>
                                                <div className="font-bold text-purple-900 text-lg">Gnina CNN</div>
                                                <div className="text-2xl font-bold text-purple-600 mt-2">{consensusResults.engines?.gnina?.best_affinity?.toFixed(1) || '-'} <span className="text-sm font-normal opacity-50">kcal/mol</span></div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* 5. Detailed Tables */}
                        <div className="space-y-6">
                            {analysis?.poses && (
                                <div className="bg-white rounded-3xl border border-slate-200 shadow-sm overflow-hidden">
                                    <div className="px-6 py-4 bg-slate-50 border-b border-slate-200 font-bold text-slate-700">Docking Poses</div>
                                    <div className="p-6"><DockingResultsTable poses={analysis.poses} /></div>
                                </div>
                            )}
                            {interactions && (
                                <div className="bg-white rounded-3xl border border-slate-200 shadow-sm overflow-hidden">
                                    <div className="px-6 py-4 bg-slate-50 border-b border-slate-200 font-bold text-slate-700">Molecular Interactions</div>
                                    <div className="p-6"><InteractionTable interactions={interactions} /></div>
                                </div>
                            )}
                            <DrugPropertiesPanel jobId={jobId} />
                        </div>

                    </div>

                    {/* RIGHT COLUMN (Viewer) - Span 5 - Sticky */}
                    <div className="lg:col-span-5 relative">
                        <div className="sticky top-28 space-y-4">
                            <div className="bg-white rounded-3xl shadow-xl border border-slate-200 overflow-hidden h-[600px] relative group">
                                <div className="absolute top-4 left-4 z-10 bg-white/90 backdrop-blur px-3 py-1 rounded-full text-xs font-bold text-slate-600 shadow-sm border border-slate-200">
                                    3D Visualization
                                </div>
                                {pdbqtData ? (
                                    <MoleculeViewer
                                        pdbqtData={pdbqtData}
                                        receptorData={receptorData}
                                        width="100%"
                                        height="100%"
                                        interactions={interactions}
                                        cavities={detectedPockets}
                                        bindingAffinity={analysis?.best_affinity || job.binding_affinity}
                                    />
                                ) : (
                                    <div className="w-full h-full flex flex-col items-center justify-center bg-slate-50 text-slate-400">
                                        <div className="animate-pulse">Loading Structure...</div>
                                    </div>
                                )}
                            </div>

                            {/* Pocket Selector Quick Access */}
                            {detectedPockets.length > 0 && (
                                <div className="bg-white p-4 rounded-2xl shadow-sm border border-slate-200">
                                    <div className="text-xs font-bold text-slate-400 uppercase mb-3">Active Pocket Overlay</div>
                                    <div className="flex flex-wrap gap-2">
                                        {detectedPockets.map(p => (
                                            <button
                                                key={p.pocket_id}
                                                onClick={() => setSelectedPocketId(p.pocket_id)}
                                                className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${selectedPocketId === p.pocket_id ? 'bg-indigo-600 text-white shadow-md' : 'bg-slate-100 text-slate-500 hover:bg-slate-200'}`}
                                            >
                                                Pocket {p.pocket_id}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                </div>
            </main>
        </div>
    )
}
