import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import MoleculeViewer from '../components/MoleculeViewer'
import ExportButtons from '../components/ExportButtons'
import { API_URL } from '../config'

export default function JobResultsPage() {
    const { jobId } = useParams()
    const [job, setJob] = useState(null)
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
        if (percentage < 30) return 'bg-blue-500'
        if (percentage < 70) return 'bg-purple-500'
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

    if (loading) return <div className="text-center py-12 text-white">Loading job details...</div>
    if (error) return <div className="text-center py-12 text-red-400">{error}</div>
    if (!job) return <div className="text-center py-12 text-white">Job not found</div>

    const progressPercentage = Math.min((elapsedTime / ESTIMATED_DURATION) * 100, 99)
    const remainingSeconds = Math.max(ESTIMATED_DURATION - elapsedTime, 0)

    return (
        <div className="min-h-screen bg-blue-mesh pt-24 pb-12">
            <main className="container mx-auto px-4">
                <div className="max-w-5xl mx-auto">
                    {/* Header */}
                    <div className="glass-modern rounded-2xl overflow-hidden mb-6">
                        <div className="bg-blue-900/50 px-8 py-6 flex justify-between items-center border-b border-blue-700/50">
                            <h1 className="text-2xl font-bold text-white">Job Details</h1>
                            <div className="flex items-center gap-4">
                                <span className="text-cyan-400 font-mono text-sm bg-blue-900/50 px-3 py-1 rounded-lg border border-blue-700/50">{job.job_id}</span>
                                <ExportButtons jobId={jobId} className="hidden md:flex" />
                            </div>
                        </div>

                        {/* Progress Bar Section */}
                        {['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(job.status) && (
                            <div className="px-8 py-6 bg-blue-900/20 border-b border-blue-700/30">
                                <div className="flex justify-between text-sm font-medium text-blue-200 mb-2">
                                    <span>Progress</span>
                                    <span>Estimated Time Remaining: {formatTime(remainingSeconds)}</span>
                                </div>
                                <div className="w-full bg-blue-900/50 rounded-full h-2.5 border border-blue-700/30">
                                    <div
                                        className={`h-2.5 rounded-full transition-all duration-500 bg-gradient-to-r from-cyan-500 to-blue-500 shadow-[0_0_10px_rgba(0,217,255,0.5)]`}
                                        style={{ width: `${progressPercentage}%` }}
                                    ></div>
                                </div>
                                <p className="text-xs text-blue-300/60 mt-2 text-center">
                                    Typical duration: 2-10 minutes. Please do not close this tab.
                                </p>
                            </div>
                        )}

                        <div className="flex items-center justify-between border-b border-blue-700/30 pb-8 px-8 pt-6">
                            <div>
                                <h2 className="text-sm font-bold text-blue-300 uppercase tracking-wide">Status</h2>
                                <div className="mt-2 flex items-center">
                                    <span className={`px-3 py-1 rounded-full text-sm font-bold border 
                        ${job.status === 'SUCCEEDED' ? 'bg-green-500/20 text-green-400 border-green-500/30' :
                                            job.status === 'FAILED' ? 'bg-red-500/20 text-red-400 border-red-500/30' :
                                                'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'}`}>
                                        {job.status}
                                    </span>
                                    {['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(job.status) && (
                                        <span className="ml-3 flex h-3 w-3 relative">
                                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                                            <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500"></span>
                                        </span>
                                    )}
                                </div>
                            </div>

                            {job.binding_affinity && (
                                <div className="text-right">
                                    <h2 className="text-sm font-bold text-blue-300 uppercase tracking-wide">Best Affinity</h2>
                                    <div className="mt-2 text-3xl font-bold text-white">{job.binding_affinity} <span className="text-lg text-blue-400 font-normal">kcal/mol</span></div>
                                </div>
                            )}
                        </div>

                        {/* 3D Molecule Viewer */}
                        {job.status === 'SUCCEEDED' && pdbqtData && (
                            <div className="p-8 border-b border-blue-700/30">
                                <MoleculeViewer
                                    pdbqtData={pdbqtData}
                                    width={700}
                                    height={500}
                                    title="Docking Result Visualization"
                                />
                            </div>
                        )}

                        {/* Downloads Section */}
                        {job.status === 'SUCCEEDED' && job.download_urls && (
                            <div className="p-8">
                                <h2 className="text-lg font-bold text-white mb-4">Results & Logs</h2>
                                <div className="grid grid-cols-2 gap-4">
                                    <a href={job.download_urls.output} className="flex items-center justify-center px-4 py-3 border border-transparent text-base font-bold rounded-xl text-blue-900 bg-cyan-400 hover:bg-cyan-300 shadow-lg shadow-cyan-500/20 transition-all">
                                        Download PDBQT Output
                                    </a>
                                    <a href={job.download_urls.log} className="flex items-center justify-center px-4 py-3 border border-blue-500/50 text-base font-bold rounded-xl text-blue-100 bg-blue-900/30 hover:bg-blue-800/50 hover:text-white transition-all">
                                        View Log File
                                    </a>
                                </div>
                            </div>
                        )}

                        {/* Failure Message */}
                        {job.status === 'FAILED' && (
                            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 m-8">
                                <h3 className="text-red-400 font-bold mb-2">Job Failed</h3>
                                <p className="text-red-200 text-sm">
                                    Please check your input files and try again. Ensure your ligand and receptor are correctly prepared.
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </main>
        </div>
    )
}
