import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'

export default function JobResultsPage() {
    const { jobId } = useParams()
    const [job, setJob] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

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
            if (!session) return

            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
            const res = await fetch(`${apiUrl}/jobs/${jobId}`, {
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                }
            })

            if (!res.ok) throw new Error('Failed to fetch job')
            const data = await res.json()
            setJob(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    if (loading) return <div className="text-center py-12">Loading job details...</div>
    if (error) return <div className="text-center py-12 text-red-600">{error}</div>
    if (!job) return <div className="text-center py-12">Job not found</div>

    return (
        <div className="min-h-screen bg-gray-50">
            <header className="bg-white shadow-sm">
                <div className="container mx-auto px-4 py-4 flex justify-between items-center">
                    <Link to="/" className="flex items-center space-x-2 text-gray-800">
                        <div className="text-2xl">🧬</div>
                        <h1 className="text-xl font-bold">CloudVina</h1>
                    </Link>
                    <Link to="/dashboard" className="text-gray-600 hover:text-purple-600 font-medium">Dashboard</Link>
                </div>
            </header>

            <main className="container mx-auto px-4 py-12">
                <div className="max-w-3xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden">
                    <div className="bg-purple-600 px-8 py-6 flex justify-between items-center">
                        <h1 className="text-2xl font-bold text-white">Job Details</h1>
                        <span className="text-purple-200 font-mono text-sm">{job.job_id}</span>
                    </div>

                    <div className="p-8 space-y-8">
                        {/* Status Section */}
                        <div className="flex items-center justify-between border-b border-gray-100 pb-8">
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
