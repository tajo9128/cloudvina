import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'

import ExportButtons from '../components/ExportButtons'
import DockingResultsTable from '../components/DockingResultsTable'
import InteractionTable from '../components/InteractionTable'
import DrugPropertiesPanel from '../components/DrugPropertiesPanel'
import { API_URL } from '../config'
import AIExplainer from '../components/AIExplainer'
import MoleculeViewer from '../components/MoleculeViewer'

export default function JobResultsPage() {
    const { jobId } = useParams()
    const [job, setJob] = useState(null)
    const [analysis, setAnalysis] = useState(null)
    const [interactions, setInteractions] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [elapsedTime, setElapsedTime] = useState(0)
    const [pdbqtData, setPdbqtData] = useState(null)

    // Multiple Pocket Results State (NEW)
    const [detectedPockets, setDetectedPockets] = useState([])
    const [selectedPocketId, setSelectedPocketId] = useState(1)

    // Consensus Results State (NEW)
    const [consensusResults, setConsensusResults] = useState(null)
    const [selectedEngine, setSelectedEngine] = useState('vina') // 'vina' or 'gnina'

    const ESTIMATED_DURATION = 300 // 5 minutes in seconds

    useEffect(() => {
        console.log('JobResultsPage mounted, checking for updates...')
        let timer
        if (job && ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(job.status)) {
            // Calculate elapsed time based on created_at if available, otherwise start from 0
            const startTime = job.created_at ? new Date(job.created_at).getTime() : Date.now()

            timer = setInterval(() => {
                const now = Date.now()
                const seconds = Math.floor((now - startTime) / 1000)
                setElapsedTime(seconds)
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
        // Poll every 5 seconds if job is running
        const interval = setInterval(() => {
            if (job && ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(job.status)) {
                fetchJob()
            }
        }, 5000)
        return () => clearInterval(interval)
    }, [jobId, job?.status])

    const fetchJob = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) throw new Error('Not authenticated')

            const res = await fetch(`${API_URL}/jobs/${jobId}`, {
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                }
            })

            if (!res.ok) throw new Error('Failed to fetch job')

            const data = await res.json()
            setJob(data)

            // Fetch analysis if job succeeded and not already fetched
            if (data.status === 'SUCCEEDED' && !analysis) {
                fetchAnalysis(session.access_token)
            }

            // Fetch interactions if job succeeded and not already fetched
            if (data.status === 'SUCCEEDED' && !interactions) {
                fetchInteractions(session.access_token)
            }

            // Fetch detected pockets for multi-pocket display
            if (data.status === 'SUCCEEDED' && detectedPockets.length === 0) {
                fetchPockets(session.access_token)
            }

            // Fetch PDBQT data for 3D viewer
            if (data.status === 'SUCCEEDED' && !pdbqtData) {
                try {
                    // For consensus, try Vina output first, fallback to standard output
                    const pdbqtUrl = data.download_urls?.output_vina || data.download_urls?.output
                    if (pdbqtUrl) {
                        const pdbqtRes = await fetch(pdbqtUrl)
                        const pdbqtText = await pdbqtRes.text()
                        setPdbqtData(pdbqtText)
                    }
                } catch (err) {
                    console.error('Failed to fetch PDBQT:', err)
                }
            }

            // Fetch Consensus Results if available
            if (data.status === 'SUCCEEDED' && data.download_urls?.results_json && !consensusResults) {
                try {
                    const consRes = await fetch(data.download_urls.results_json)
                    const consData = await consRes.json()
                    setConsensusResults(consData)
                } catch (err) {
                    console.error('Failed to fetch consensus results:', err)
                }
            }
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const fetchAnalysis = async (token) => {
        try {
            const res = await fetch(`${API_URL}/jobs/${jobId}/analyze`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            })

            if (res.ok) {
                const data = await res.json()
                setAnalysis(data.analysis)
            } else {
                // Set empty analysis to prevent retrying
                console.error('Analysis fetch failed:', res.status)
                setAnalysis({ error: true })
            }
        } catch (err) {
            console.error('Failed to fetch analysis:', err)
            setAnalysis({ error: true })
        }
    }

    const fetchInteractions = async (token) => {
        try {
            const res = await fetch(`${API_URL}/jobs/${jobId}/interactions`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            })

            if (res.ok) {
                const data = await res.json()
                setInteractions(data.interactions)
            } else {
                console.error('Interactions fetch failed:', res.status)
                setInteractions({ error: true })
            }
        } catch (err) {
            console.error('Failed to fetch interactions:', err)
            setInteractions({ error: true })
        }
    }

    // Fetch detected pockets for multi-pocket display
    const fetchPockets = async (token) => {
        try {
            const res = await fetch(`${API_URL}/jobs/${jobId}/detect-cavities`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            })

            if (res.ok) {
                const data = await res.json()
                if (data.cavities && data.cavities.length > 0) {
                    setDetectedPockets(data.cavities)
                    console.log('Detected pockets:', data.cavities.length)
                }
            }
        } catch (err) {
            console.error('Failed to fetch pockets:', err)
        }
    }

    if (loading) return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center">
            <div className="text-center">
                <div className="inline-block p-4 rounded-full bg-primary-50 text-primary-600 mb-4">
                    <svg className="w-8 h-8 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                </div>
                <div className="text-slate-500 font-medium">Loading job details...</div>
            </div>
        </div>
    )

    if (error) return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center">
            <div className="text-center max-w-md mx-auto px-4">
                <div className="bg-red-50 text-red-600 p-6 rounded-xl border border-red-100 mb-6">
                    <svg className="w-12 h-12 mx-auto mb-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    <h3 className="text-lg font-bold mb-2">Error Loading Job</h3>
                    <p>{error}</p>
                </div>
                <Link to="/dashboard" className="btn-primary">Return to Dashboard</Link>
            </div>
        </div>
    )

    if (!job) return <div className="text-center py-12 text-slate-500">Job not found</div>

    const progressPercentage = Math.min((elapsedTime / ESTIMATED_DURATION) * 100, 99)
    const remainingSeconds = Math.max(ESTIMATED_DURATION - elapsedTime, 0)

    return (
        <div className="min-h-screen bg-slate-50 pt-24 pb-12">
            <main className="container mx-auto px-4">
                <div className="max-w-7xl mx-auto">
                    {/* Header */}
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden mb-6">
                        <div className="bg-slate-50 px-8 py-6 flex flex-col md:flex-row justify-between items-start md:items-center border-b border-slate-200 gap-4">
                            <div>
                                <div className="flex items-center gap-3 mb-1">
                                    <h1 className="text-2xl font-bold text-slate-900">Job Details</h1>
                                    <span className="px-3 py-1 rounded-full text-xs font-bold bg-slate-200 text-slate-600 font-mono">
                                        {job.job_id.slice(0, 8)}
                                    </span>
                                </div>
                                <p className="text-slate-500 text-sm">Created on {new Date(job.created_at).toLocaleString()}</p>
                            </div>
                            <div className="flex items-center gap-4">
                                <ExportButtons jobId={jobId} className="hidden md:flex" />
                                <Link to="/dashboard" className="text-slate-500 hover:text-primary-600 font-medium text-sm">
                                    &larr; Back to Dashboard
                                </Link>
                            </div>
                        </div>

                        {/* Progress Bar Section */}
                        {['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(job.status) && (
                            <div className="px-8 py-8 bg-primary-50/30 border-b border-slate-200">
                                <div className="flex justify-between text-sm font-bold text-slate-700 mb-3">
                                    <span className="flex items-center gap-2">
                                        <span className="relative flex h-3 w-3">
                                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary-400 opacity-75"></span>
                                            <span className="relative inline-flex rounded-full h-3 w-3 bg-primary-500"></span>
                                        </span>
                                        Simulation in Progress
                                    </span>
                                    <span className="font-mono text-slate-500">Est. Remaining: {formatTime(remainingSeconds)}</span>
                                </div>
                                <div className="w-full bg-slate-200 rounded-full h-3 border border-slate-300/50 overflow-hidden">
                                    <div
                                        className={`h-full rounded-full transition-all duration-500 bg-gradient-to-r from-primary-500 to-secondary-500`}
                                        style={{ width: `${progressPercentage}%` }}
                                    >
                                        <div className="w-full h-full animate-pulse bg-white/30"></div>
                                    </div>
                                </div>
                                <p className="text-xs text-slate-500 mt-3 text-center">
                                    Typical duration: 2-10 minutes. You can safely close this tab and check back later.
                                </p>
                            </div>
                        )}
                    </div>

                    {/* Split Layout */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Left Column: Data & Analysis */}
                        <div className="space-y-6">
                            {/* Compact Job Details Card */}
                            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                                <div className="flex justify-between items-start mb-4">
                                    <h2 className="text-lg font-bold text-slate-900">Job Details</h2>
                                    <span className={`px-3 py-1 rounded-full text-sm font-bold border ${job.status === 'SUCCEEDED' ? 'bg-green-100 text-green-700 border-green-200' : job.status === 'FAILED' ? 'bg-red-100 text-red-700 border-red-200' : 'bg-amber-100 text-amber-700 border-amber-200'}`}>
                                        {job.status}
                                    </span>
                                </div>

                                <div className="grid grid-cols-2 gap-3 text-sm mb-4">
                                    <div>
                                        <span className="font-semibold text-slate-600">Job ID:</span>
                                        <span className="ml-2 text-slate-900">{jobId.slice(0, 8)}...</span>
                                    </div>
                                    <div>
                                        <span className="font-semibold text-slate-600">Mode:</span>
                                        <span className="ml-2 text-slate-900">{consensusResults ? 'Consensus' : 'Single Engine'}</span>
                                    </div>
                                    <div>
                                        <span className="font-semibold text-slate-600">Receptor:</span>
                                        <span className="ml-2 text-slate-900">{job.receptor_filename || 'Unknown'}</span>
                                    </div>
                                    <div>
                                        <span className="font-semibold text-slate-600">Ligand:</span>
                                        <span className="ml-2 text-slate-900">{job.ligand_filename || 'Unknown'}</span>
                                    </div>
                                    <div className="col-span-2">
                                        <span className="font-semibold text-slate-600">Created:</span>
                                        <span className="ml-2 text-slate-900">{new Date(job.created_at).toLocaleString()}</span>
                                    </div>
                                </div>

                                {/* Best Affinity - Consensus or Single */}
                                <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-4 mb-4 border border-green-200">
                                    <div className="text-center">
                                        <p className="text-xs font-semibold text-green-700 uppercase mb-1">Best Affinity</p>
                                        {consensusResults ? (
                                            <>
                                                <p className="text-4xl font-bold text-green-800">
                                                    {consensusResults.best_affinity ? consensusResults.best_affinity.toFixed(2) : '-'}
                                                </p>
                                                <p className="text-sm text-green-700 mt-1">kcal/mol</p>
                                                <p className="text-xs text-green-600 mt-2">
                                                    Vina: {consensusResults.engines?.vina?.best_affinity?.toFixed(2) || 'N/A'} |
                                                    Gnina: {consensusResults.engines?.gnina?.best_affinity?.toFixed(2) || 'N/A'}
                                                </p>
                                            </>
                                        ) : (
                                            <>
                                                <p className="text-4xl font-bold text-green-800">
                                                    {analysis?.best_affinity || '-'}
                                                </p>
                                                <p className="text-sm text-green-700 mt-1">kcal/mol</p>
                                            </>
                                        )}
                                    </div>
                                </div>

                                {/* Download Reports Section */}
                                {job.status === 'SUCCEEDED' && (
                                    <div className="pt-4 border-t border-slate-200">
                                        <h3 className="text-sm font-semibold text-slate-700 mb-3">📥 Download Reports</h3>
                                        <div className="flex flex-wrap gap-2">
                                            {job.download_urls?.results_csv && (
                                                <a
                                                    href={job.download_urls.results_csv}
                                                    download
                                                    className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
                                                >
                                                    📊 CSV Report
                                                </a>
                                            )}
                                            <a
                                                href={`${API_URL}/jobs/${jobId}/export/pdf`}
                                                className="inline-flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm font-medium"
                                            >
                                                📄 PDF Report
                                            </a>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Consensus Results Card (NEW) */}
                            {consensusResults && (
                                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden mb-6 p-6">
                                    <div className="flex items-center justify-between mb-4">
                                        <h3 className="font-bold text-slate-900 flex items-center gap-2">
                                            <span className="text-xl">📊</span> Consensus Docking Report
                                        </h3>
                                        <div className="text-sm font-bold text-slate-500">
                                            Average Affinity: <span className="text-green-600 text-lg">{consensusResults.average_affinity?.toFixed(2)} kcal/mol</span>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                                        {/* Vina Result */}
                                        {consensusResults.engines?.vina && (
                                            <div className="p-4 bg-blue-50 rounded-xl border border-blue-100 flex flex-col items-center">
                                                <div className="text-2xl mb-1">🐢</div>
                                                <div className="font-bold text-blue-900">AutoDock Vina</div>
                                                <div className="text-2xl font-bold text-blue-700 mt-2">
                                                    {consensusResults.engines.vina.best_affinity?.toFixed(1) || 'N/A'}
                                                </div>
                                                <div className="text-xs text-blue-500">kcal/mol</div>
                                            </div>
                                        )}

                                        {/* rDock Result */}
                                        {consensusResults.engines?.rdock && (
                                            <div className="p-4 bg-orange-50 rounded-xl border border-orange-100 flex flex-col items-center">
                                                <div className="text-2xl mb-1">⚡</div>
                                                <div className="font-bold text-orange-900">rDock</div>
                                                <div className="text-2xl font-bold text-orange-700 mt-2">
                                                    {consensusResults.engines.rdock.best_affinity?.toFixed(1) || 'N/A'}
                                                </div>
                                                <div className="text-xs text-orange-500">rDock Score</div>
                                            </div>
                                        )}

                                        {/* Gnina Result */}
                                        {consensusResults.engines?.gnina && (
                                            <div className="p-4 bg-purple-50 rounded-xl border border-purple-100 flex flex-col items-center">
                                                <div className="text-2xl mb-1">🧠</div>
                                                <div className="font-bold text-purple-900">Gnina (AI)</div>
                                                <div className="text-2xl font-bold text-purple-700 mt-2">
                                                    {consensusResults.engines.gnina.best_affinity?.toFixed(1) || 'N/A'}
                                                </div>
                                                <div className="text-xs text-purple-500">kcal/mol</div>
                                                {consensusResults.engines.gnina.cnn_score && (
                                                    <div className="mt-1 px-2 py-0.5 bg-purple-200 text-purple-800 rounded text-xs font-bold">
                                                        CNN: {consensusResults.engines.gnina.cnn_score}
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Multiple Pocket Selector (NEW) */}
                            {job.status === 'SUCCEEDED' && detectedPockets.length > 1 && (
                                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden p-6">
                                    <div className="flex items-center justify-between mb-4">
                                        <div>
                                            <h3 className="font-bold text-slate-900">Binding Pockets Detected</h3>
                                            <p className="text-sm text-slate-500">
                                                {detectedPockets.length} potential binding sites identified
                                            </p>
                                        </div>
                                    </div>

                                    <div className="flex flex-wrap gap-2">
                                        {detectedPockets.map((pocket) => (
                                            <button
                                                key={pocket.pocket_id}
                                                onClick={() => setSelectedPocketId(pocket.pocket_id)}
                                                className={`px-4 py-2 rounded-lg font-medium transition-all ${selectedPocketId === pocket.pocket_id
                                                    ? 'bg-primary-600 text-white shadow-md'
                                                    : 'bg-white text-slate-700 border border-slate-200 hover:border-primary-300'
                                                    }`}
                                            >
                                                <div className="flex items-center gap-2">
                                                    <span>Pocket {pocket.pocket_id}</span>
                                                    <span className={`text-xs px-1.5 py-0.5 rounded ${selectedPocketId === pocket.pocket_id
                                                        ? 'bg-white/20'
                                                        : 'bg-primary-100 text-primary-700'
                                                        }`}>
                                                        {(pocket.score * 100).toFixed(0)}%
                                                    </span>
                                                </div>
                                            </button>
                                        ))}
                                    </div>

                                    {/* Pocket Details */}
                                    {detectedPockets.find(p => p.pocket_id === selectedPocketId) && (
                                        <div className="mt-4 p-4 bg-slate-50 rounded-lg border border-slate-200">
                                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                                                <div>
                                                    <div className="text-xs text-slate-500 uppercase">Center X</div>
                                                    <div className="font-mono font-bold text-slate-900">
                                                        {detectedPockets.find(p => p.pocket_id === selectedPocketId).center_x}
                                                    </div>
                                                </div>
                                                <div>
                                                    <div className="text-xs text-slate-500 uppercase">Center Y</div>
                                                    <div className="font-mono font-bold text-slate-900">
                                                        {detectedPockets.find(p => p.pocket_id === selectedPocketId).center_y}
                                                    </div>
                                                </div>
                                                <div>
                                                    <div className="text-xs text-slate-500 uppercase">Center Z</div>
                                                    <div className="font-mono font-bold text-slate-900">
                                                        {detectedPockets.find(p => p.pocket_id === selectedPocketId).center_z}
                                                    </div>
                                                </div>
                                                <div>
                                                    <div className="text-xs text-slate-500 uppercase">Volume</div>
                                                    <div className="font-mono font-bold text-slate-900">
                                                        {detectedPockets.find(p => p.pocket_id === selectedPocketId).volume?.toFixed(0)} Å³
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Docking Results Table */}
                            {job.status === 'SUCCEEDED' && analysis && !analysis.error && analysis.poses && (
                                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                                    <div className="p-6 border-b border-slate-200 bg-slate-50">
                                        <h3 className="font-bold text-slate-900">Docking Poses</h3>
                                    </div>
                                    <div className="p-6">
                                        <DockingResultsTable poses={analysis.poses} />
                                    </div>
                                </div>
                            )}

                            {/* Interaction Analysis Table */}
                            {job.status === 'SUCCEEDED' && interactions && !interactions.error && (
                                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                                    <div className="p-6 border-b border-slate-200 bg-slate-50">
                                        <h3 className="font-bold text-slate-900">Molecular Interactions</h3>
                                    </div>
                                    <div className="p-6">
                                        <InteractionTable interactions={interactions} />
                                    </div>
                                </div>
                            )}

                            {/* Drug Properties Panel (NEW) */}
                            {job.status === 'SUCCEEDED' && (
                                <DrugPropertiesPanel jobId={jobId} />
                            )}

                            {/* AI Explainer */}
                            {job.status === 'SUCCEEDED' && (
                                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                                    <AIExplainer jobId={jobId} />
                                </div>
                            )}

                            {/* Downloads Section */}
                            {job.status === 'SUCCEEDED' && job.download_urls && (
                                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden p-8">
                                    <h2 className="text-lg font-bold text-slate-900 mb-6 flex items-center gap-2">
                                        <svg className="w-5 h-5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                                        Downloads & Data
                                    </h2>
                                    <div className="grid sm:grid-cols-2 gap-4">
                                        {/* Consensus Mode - Separate Vina and Gnina Downloads */}
                                        {consensusResults ? (
                                            <>
                                                {job.download_urls.output_vina && (
                                                    <a href={job.download_urls.output_vina} className="group flex items-center justify-between px-6 py-4 border border-blue-200 rounded-xl bg-blue-50 text-blue-700 hover:bg-blue-100 hover:border-blue-300 transition-all">
                                                        <div className="flex items-center gap-3">
                                                            <div className="w-10 h-10 rounded-lg bg-white flex items-center justify-center text-blue-600 shadow-sm">
                                                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
                                                            </div>
                                                            <div className="text-left">
                                                                <div className="font-bold">Vina Structure</div>
                                                                <div className="text-xs opacity-75">PDBQT Format</div>
                                                            </div>
                                                        </div>
                                                        <svg className="w-5 h-5 transform group-hover:translate-y-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                                                    </a>
                                                )}
                                                {job.download_urls.output_gnina && (
                                                    <a href={job.download_urls.output_gnina} className="group flex items-center justify-between px-6 py-4 border border-purple-200 rounded-xl bg-purple-50 text-purple-700 hover:bg-purple-100 hover:border-purple-300 transition-all">
                                                        <div className="flex items-center gap-3">
                                                            <div className="w-10 h-10 rounded-lg bg-white flex items-center justify-center text-purple-600 shadow-sm">
                                                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>
                                                            </div>
                                                            <div className="text-left">
                                                                <div className="font-bold">Gnina Structure</div>
                                                                <div className="text-xs opacity-75">AI-Generated PDBQT</div>
                                                            </div>
                                                        </div>
                                                        <svg className="w-5 h-5 transform group-hover:translate-y-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                                                    </a>
                                                )}
                                            </>
                                        ) : (
                                            /* Single Mode - Original Download */
                                            <a href={job.download_urls.output} className="group flex items-center justify-between px-6 py-4 border border-primary-200 rounded-xl bg-primary-50 text-primary-700 hover:bg-primary-100 hover:border-primary-300 transition-all">
                                                <div className="flex items-center gap-3">
                                                    <div className="w-10 h-10 rounded-lg bg-white flex items-center justify-center text-primary-600 shadow-sm">
                                                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
                                                    </div>
                                                    <div className="text-left">
                                                        <div className="font-bold">Docked Structure</div>
                                                        <div className="text-xs opacity-75">PDBQT Format</div>
                                                    </div>
                                                </div>
                                                <svg className="w-5 h-5 transform group-hover:translate-y-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                                            </a>
                                        )}

                                        <a href={job.download_urls.log} className="group flex items-center justify-between px-6 py-4 border border-slate-200 rounded-xl bg-white text-slate-700 hover:bg-slate-50 hover:border-slate-300 transition-all">
                                            <div className="flex items-center gap-3">
                                                <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center text-slate-500 shadow-sm">
                                                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                                                </div>
                                                <div className="text-left">
                                                    <div className="font-bold">Execution Log</div>
                                                    <div className="text-xs opacity-75">Text Format</div>
                                                </div>
                                            </div>
                                            <svg className="w-5 h-5 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>
                                        </a>

                                        {job.download_urls.config && (
                                            <a href={job.download_urls.config} className="group flex items-center justify-between px-6 py-4 border border-green-200 rounded-xl bg-green-50 text-green-700 hover:bg-green-100 hover:border-green-300 transition-all">
                                                <div className="flex items-center gap-3">
                                                    <div className="w-10 h-10 rounded-lg bg-white flex items-center justify-center text-green-600 shadow-sm">
                                                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                                                    </div>
                                                    <div className="text-left">
                                                        <div className="font-bold">Config File</div>
                                                        <div className="text-xs opacity-75">Vina Parameters</div>
                                                    </div>
                                                </div>
                                                <svg className="w-5 h-5 transform group-hover:translate-y-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                                            </a>
                                        )}

                                        {/* CSV Export Button (New) */}
                                        {job.download_urls.results_csv && (
                                            <a href={job.download_urls.results_csv} className="group flex items-center justify-between px-6 py-4 border border-violet-200 rounded-xl bg-violet-50 text-violet-700 hover:bg-violet-100 hover:border-violet-300 transition-all">
                                                <div className="flex items-center gap-3">
                                                    <div className="w-10 h-10 rounded-lg bg-white flex items-center justify-center text-violet-600 shadow-sm">
                                                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                                                    </div>
                                                    <div className="text-left">
                                                        <div className="font-bold">Summary Report</div>
                                                        <div className="text-xs opacity-75">CSV Format</div>
                                                    </div>
                                                </div>
                                                <svg className="w-5 h-5 transform group-hover:translate-y-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                                            </a>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Failure Message */}
                            {job.status === 'FAILED' && (
                                <div className="bg-red-50 border-t border-red-100 p-8 rounded-2xl">
                                    <div className="flex items-start gap-4">
                                        <div className="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center text-red-600 flex-shrink-0">
                                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                                        </div>
                                        <div>
                                            <h3 className="text-red-800 font-bold text-lg mb-2">Simulation Failed</h3>
                                            <p className="text-red-600 mb-4">
                                                The docking simulation could not be completed. This is usually due to issues with the input files (e.g., incorrect format, missing atoms) or system constraints.
                                            </p>
                                            <div className="bg-white border border-red-200 rounded-lg p-4 text-sm font-mono text-red-700 overflow-x-auto">
                                                {job.error_message || "No specific error details available."}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Right Column: Sticky 3D Viewer */}
                        <div className="relative">
                            <div className="sticky top-24">
                                {pdbqtData ? (
                                    <div className="bg-white rounded-2xl shadow-lg border border-slate-200 overflow-hidden">
                                        <div className="p-4 bg-slate-50 border-b border-slate-200 flex justify-between items-center">
                                            <h3 className="font-bold text-slate-900">3D Visualization</h3>
                                            <Link to={`/3d-viewer/${jobId}`} className="text-xs text-primary-600 hover:text-primary-700 font-medium flex items-center gap-1">
                                                Full Screen
                                                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>
                                            </Link>
                                        </div>
                                        <div className="h-[600px] w-full relative">
                                            {(pdbqtData && pdbqtData.trim()) ? (
                                                <MoleculeViewer
                                                    pdbqtData={pdbqtData}
                                                    width="100%"
                                                    height="100%"
                                                    title=""
                                                    interactions={interactions}
                                                    cavities={detectedPockets}
                                                />
                                            ) : (
                                                <div className="flex items-center justify-center h-full text-slate-400">
                                                    Unable to load structure data.
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="bg-slate-50 rounded-2xl border border-slate-200 p-12 text-center h-[600px] flex flex-col items-center justify-center">
                                        <div className="w-16 h-16 bg-slate-200 rounded-full flex items-center justify-center mb-4 text-slate-400">
                                            <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
                                        </div>
                                        <h3 className="text-slate-900 font-bold mb-2">3D Viewer</h3>
                                        <p className="text-slate-500 text-sm max-w-xs mx-auto">
                                            {job.status === 'SUCCEEDED'
                                                ? "Loading structure..."
                                                : "Visualization will be available once the simulation completes."}
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </main>

        </div>
    )
}
