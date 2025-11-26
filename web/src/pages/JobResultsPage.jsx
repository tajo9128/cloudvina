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

    if (loading) return <div className="text-center py-12">Loading job details...</div>
    if (error) return <div className="text-center py-12 text-red-600">{error}</div>
    if (!job) return <div className="text-center py-12">Job not found</div>

    const progressPercentage = Math.min((elapsedTime / ESTIMATED_DURATION) * 100, 99)
    const remainingSeconds = Math.max(ESTIMATED_DURATION - elapsedTime, 0)

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50">

            <main className="container mx-auto px-4 py-12">
                <div className="max-w-5xl mx-auto">
                    {/* Header */}
                    <div className="bg-white rounded-xl shadow-lg overflow-hidden mb-6">
                        <div className="bg-purple-600 px-8 py-6 flex justify-between items-center">
                            <h1 className="text-2xl font-bold text-white">Job Details</h1>
                            <div className="flex items-center gap-4">
                                <span className="text-purple-200 font-mono text-sm">{job.job_id}</span>
                                <ExportButtons jobId={jobId} className="hidden md:flex" />
                            </div>
                        </div>

                        {/* Progress Bar Section */}
                        {['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(job.status) && (
                            <div className="px-8 py-6 bg-gray-50 border-b border-gray-100">
                                <div className="flex justify-between text-sm font-medium text-gray-600 mb-2">
                                    <span>Progress</span>
                                    <span>Estimated Time Remaining: {formatTime(remainingSeconds)}</span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-2.5">
                                    <div
                                        className={`h-2.5 rounded-full transition-all duration-500 ${getProgressColor(progressPercentage)}`}
                                        style={{ width: `${progressPercentage}%` }}
                                    ></div>
                                </div>
                                <p className="text-xs text-gray-400 mt-2 text-center">
                                    Typical duration: 2-10 minutes. Please do not close this tab.
                                </p>
                            </div>
                        )}

                        <div className="flex items-center justify-between border-b border-gray-100 pb-8 px-8 pt-6">
                            <div>
                                <h2 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Status</h2>
                                <div className="mt-2 flex items-center">
                                    <span className={`px-3 py-1 rounded-full text-sm font-bold 
                        ${job.status === 'SUCCEEDED' ? 'bg-green-100 text-green-800' :
                                            job.status === 'FAILED' ? 'bg-red-100 text-red-800' :
                                                'bg-yellow-100 text-yellow-800'}`}>
                                        {job.status}
                                    </span>
                                    {['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(job.status) && (
                                        <span className="ml-3 flex h-3 w-3 relative">
                                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
                                            <span className="relative inline-flex rounded-full h-3 w-3 bg-purple-500"></span>
                                        </span>
                                    )}
                                </div>
                            </div>

                            {job.binding_affinity && (
                                <div className="text-right">
                                    <h2 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Best Affinity</h2>
                                    <div className="mt-2 text-3xl font-bold text-gray-900">{job.binding_affinity} <span className="text-lg text-gray-400 font-normal">kcal/mol</span></div>
                                </div>
                            )}
                        </div>

                        {/* 3D Molecule Viewer */}
                        {job.status === 'SUCCEEDED' && pdbqtData && (
                            <div>
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
                            <div>
                                <h2 className="text-lg font-bold text-gray-900 mb-4">Results & Logs</h2>
                                <div className="grid grid-cols-2 gap-4">
                                    <a href={job.download_urls.output} className="flex items-center justify-center px-4 py-3 border border-transparent text-base font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700 shadow-sm">
                                        Download PDBQT Output
                                    </a>
                                    <a href={job.download_urls.log} className="flex items-center justify-center px-4 py-3 border border-gray-300 text-base font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 shadow-sm">
                                        View Log File
                                    </a>
                                </div>
                            </div>
                        )}

                        {/* Failure Message */}
                        {job.status === 'FAILED' && (
                            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                                <h3 className="text-red-800 font-bold mb-2">Job Failed</h3>
                                <p className="text-red-600 text-sm">
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
