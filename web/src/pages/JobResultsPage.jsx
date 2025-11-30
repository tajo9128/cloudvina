import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import MoleculeViewer from '../components/MoleculeViewer'
import ExportButtons from '../components/ExportButtons'
import DockingResultsTable from '../components/DockingResultsTable'
import InteractionTable from '../components/InteractionTable'
import { API_URL } from '../config'

export default function JobResultsPage() {
    const { jobId } = useParams()
    const [job, setJob] = useState(null)
    const [analysis, setAnalysis] = useState(null)
    const [interactions, setInteractions] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [elapsedTime, setElapsedTime] = useState(0)
    const [pdbqtData, setPdbqtData] = useState(null)
    const ESTIMATED_DURATION = 300 // 5 minutes in seconds

    useEffect(() => {
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

    const getProgressColor = (percentage) => {
        if (percentage < 30) return 'bg-primary-500'
        if (percentage < 70) return 'bg-secondary-500'
        return 'bg-green-500'
    }

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

            // Fetch PDBQT data if succeeded
            if (data.status === 'SUCCEEDED' && data.download_urls?.output && !pdbqtData) {
                try {
                    const pdbqtRes = await fetch(data.download_urls.output)
                    const pdbqtText = await pdbqtRes.text()
                    setPdbqtData(pdbqtText)
                } catch (err) {
                    console.error('Failed to fetch PDBQT:', err)
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
            const res = await fetch(`${API_URL}/jobs/${jobId}/analysis`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            })

            if (res.ok) {
                const data = await res.json()
                setAnalysis(data.analysis)
            }
        } catch (err) {
            console.error('Failed to fetch analysis:', err)
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
            }
        } catch (err) {
            console.error('Failed to fetch interactions:', err)
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
                <div className="max-w-6xl mx-auto">
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

                        <div className="grid md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-slate-200 border-b border-slate-200">
                            <div className="p-6 text-center">
                                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Status</h2>
                                <div className="inline-flex items-center">
                                    <span className={`px-4 py-1.5 rounded-full text-sm font-bold border 
                        ${job.status === 'SUCCEEDED' ? 'bg-green-100 text-green-700 border-green-200' :
                                            job.status === 'FAILED' ? 'bg-red-100 text-red-700 border-red-200' :
                                                'bg-amber-100 text-amber-700 border-amber-200'}`}>
                                        {job.status}
                                    </span>
                                </div>
                            </div>

                            <div className="p-6 text-center">
                                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Best Affinity</h2>
                                {analysis?.best_affinity ? (
                                    <>
                                        <div className="text-3xl font-bold text-green-700">{analysis.best_affinity} <span className="text-sm text-slate-500 font-normal">kcal/mol</span></div>
                                        {analysis.num_poses > 1 && (
                                            <div className="text-xs text-slate-500 mt-2">
                                                Range: {analysis.energy_range_min} to {analysis.energy_range_max}
                                            </div>
                                        )}
                                    </>
                                ) : (
                                    <div className="text-3xl font-bold text-slate-300">-</div>
                                )}
                            </div>

                            <div className="p-6 text-center">
                                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Ligand</h2>
                                <div className="text-lg font-medium text-slate-700 truncate px-4" title={job.ligand_filename || 'Unknown'}>
                                    {job.ligand_filename || 'Unknown'}
                                </div>
                            </div>
                        </div>

                        {/* 3D Molecule Viewer */}
                        {job.status === 'SUCCEEDED' && pdbqtData && (
                            <div className="p-8 border-b border-slate-200 bg-slate-50/50">
                                <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                                    <div className="p-4 border-b border-slate-100 flex justify-between items-center">
                                        <h3 className="font-bold text-slate-700">3D Visualization</h3>
                                        <span className="text-xs text-slate-400">Powered by NGL Viewer</span>
                                    </div>
                                    <div className="h-[500px] w-full">
                                        <MoleculeViewer
                                            pdbqtData={pdbqtData}
                                            width="100%"
                                            height="100%"
                                            title="Docking Result Visualization"
                                        />
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Docking Results Table */}
                        {job.status === 'SUCCEEDED' && analysis?.poses && (
                            <div className="p-8 bg-white">
                                <DockingResultsTable poses={analysis.poses} />
                            </div>
                        )}

                        {/* Interaction Analysis Table */}
                        {job.status === 'SUCCEEDED' && interactions && (
                            <div className="p-8 bg-white border-t border-slate-100">
                                <InteractionTable interactions={interactions} />
                            </div>
                        )}

                        {/* Downloads Section */}
                        {job.status === 'SUCCEEDED' && job.download_urls && (
                            <div className="p-8 bg-white">
                                <h2 className="text-lg font-bold text-slate-900 mb-6 flex items-center gap-2">
                                    <svg className="w-5 h-5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                                    Downloads & Data
                                </h2>
                                <div className="grid sm:grid-cols-2 gap-4">
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
                                </div>
                            </div>
                        )}

                        {/* Failure Message */}
                        {job.status === 'FAILED' && (
                            <div className="bg-red-50 border-t border-red-100 p-8">
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
                </div>
            </main>
        </div>
    )
}
