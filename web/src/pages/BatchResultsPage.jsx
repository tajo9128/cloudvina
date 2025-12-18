import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'
import MoleculeViewer from '../components/MoleculeViewer'

export default function BatchResultsPage() {
    const { batchId } = useParams() // Note: Route might use :jobId or :batchId, handling both below

    const [batchData, setBatchData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [sortConfig, setSortConfig] = useState({ key: 'binding_affinity', direction: 'ascending' })

    // State for First Job Visualization
    const [firstJobPdbqt, setFirstJobPdbqt] = useState(null)
    const [firstJobReceptor, setFirstJobReceptor] = useState(null)
    const [firstJobId, setFirstJobId] = useState(null)
    const [elapsedTime, setElapsedTime] = useState(0)

    const ESTIMATED_DURATION = 600 // 10 mins estimate for batches

    useEffect(() => {
        if (batchId) {
            fetchBatchDetails(batchId)
        }
    }, [batchId])

    useEffect(() => {
        let timer
        if (batchData && ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(batchData.status)) {
            const startTime = batchData.created_at ? new Date(batchData.created_at).getTime() : Date.now()
            timer = setInterval(() => {
                const now = Date.now()
                const seconds = Math.floor((now - startTime) / 1000)
                setElapsedTime(seconds)
            }, 1000)
        }
        return () => clearInterval(timer)
    }, [batchData])

    // Poll for updates if running
    useEffect(() => {
        const interval = setInterval(() => {
            if (batchData && ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(batchData.status)) {
                fetchBatchDetails(batchId)
            }
        }, 5000)
        return () => clearInterval(interval)
    }, [batchId, batchData?.status])


    const fetchBatchDetails = async (id) => {
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) return

            const response = await fetch(`${API_URL}/jobs/batch/${id}`, {
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                }
            })

            if (!response.ok) throw new Error('Failed to fetch batch details')

            const data = await response.json()
            setBatchData(data)

            // Logic to fetch Best Job Data for 3D Viewer
            if (data.jobs && data.jobs.length > 0) {
                const validJobs = data.jobs.filter(j => j.status === 'SUCCEEDED' && j.binding_affinity !== null)
                if (validJobs.length > 0) {
                    // Sort by affinity (negative is better)
                    const bestJob = validJobs.sort((a, b) => a.binding_affinity - b.binding_affinity)[0]

                    // Only fetch if different
                    if (bestJob.id !== firstJobId) {
                        setFirstJobId(bestJob.id)
                        fetchJobStructure(bestJob.id, session.access_token)
                    }
                }
            }

        } catch (err) {
            console.error(err)
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const fetchJobStructure = async (jobId, token) => {
        try {
            const res = await fetch(`${API_URL}/jobs/${jobId}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            })
            if (!res.ok) return
            const jobData = await res.json()

            // Output URL
            if (jobData.download_urls?.output_vina || jobData.download_urls?.output) {
                const url = jobData.download_urls.output_vina || jobData.download_urls.output
                const pdbqtRes = await fetch(url)
                const text = await pdbqtRes.text()
                setFirstJobPdbqt(text)
            }
            // Receptor URL
            if (jobData.download_urls?.receptor) {
                const recRes = await fetch(jobData.download_urls.receptor)
                const text = await recRes.text()
                setFirstJobReceptor(text)
            }
        } catch (e) {
            console.error("Failed to load 3D structure for batch top hit", e)
        }
    }

    const handleSort = (key) => {
        let direction = 'ascending'
        if (sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending'
        }
        setSortConfig({ key, direction })
    }

    const sortedJobs = batchData?.jobs ? [...batchData.jobs].sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) {
            return sortConfig.direction === 'ascending' ? -1 : 1
        }
        if (a[sortConfig.key] > b[sortConfig.key]) {
            return sortConfig.direction === 'ascending' ? 1 : -1
        }
        return 0
    }) : []

    const downloadCSV = () => {
        if (!batchData?.jobs) return
        const headers = ['Ligand Name', 'Status', 'Binding Affinity (kcal/mol)', 'Job ID']
        const rows = batchData.jobs.map(job => [
            job.ligand_filename.replace('.pdbqt', ''),
            job.status,
            job.binding_affinity || 'N/A',
            job.id
        ])
        const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\n')
        const blob = new Blob([csvContent], { type: 'text/csv' })
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `batch_results_${batchData.batch_id}.csv`
        a.click()
        window.URL.revokeObjectURL(url)
    }

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}m ${secs}s`
    }

    if (loading) return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50">
            <div className="text-center">
                <div className="inline-block p-4 rounded-full bg-primary-50 text-primary-600 mb-4">
                    <svg className="w-8 h-8 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                </div>
                <div className="text-slate-500">Loading Batch Analysis...</div>
            </div>
        </div>
    )

    if (error || !batchData) return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50">
            <div className="text-center max-w-md">
                <div className="text-4xl mb-4">⚠️</div>
                <h2 className="text-xl font-bold text-slate-900">Batch Not Found</h2>
                <p className="text-slate-500 mt-2">{error || "The requested batch job does not exist."}</p>
                <Link to="/dashboard" className="btn-secondary mt-4 inline-flex">Return to Dashboard</Link>
            </div>
        </div>
    )

    const progressPercentage = Math.min((elapsedTime / ESTIMATED_DURATION) * 100, 99)
    const remainingSeconds = Math.max(ESTIMATED_DURATION - elapsedTime, 0)

    return (
        <div className="min-h-screen bg-slate-50 pt-24 pb-12">
            <main className="container mx-auto px-4">
                <div className="max-w-7xl mx-auto">
                    {/* Header Card (Unified Style) */}
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden mb-6">
                        <div className="bg-slate-50 px-8 py-6 flex flex-col md:flex-row justify-between items-start md:items-center border-b border-slate-200 gap-4">
                            <div>
                                <div className="flex items-center gap-3 mb-1">
                                    <h1 className="text-2xl font-bold text-slate-900">Batch Results</h1>
                                    <span className="px-3 py-1 rounded-full text-xs font-bold bg-slate-200 text-slate-600 font-mono">
                                        {batchData.batch_id.slice(0, 8)}
                                    </span>
                                </div>
                                <p className="text-slate-500 text-sm">Started {new Date(batchData.created_at).toLocaleString()}</p>
                            </div>
                            <div className="flex items-center gap-4">
                                <button onClick={downloadCSV} className="text-primary-600 hover:text-primary-700 font-medium text-sm flex items-center gap-1">
                                    <span>⬇️</span> Export CSV
                                </button>
                                <Link to="/dashboard" className="text-slate-500 hover:text-primary-600 font-medium text-sm">
                                    &larr; Back to Dashboard
                                </Link>
                            </div>
                        </div>

                        {/* Progress Section */}
                        {['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(batchData.status) && (
                            <div className="px-8 py-8 bg-primary-50/30 border-b border-slate-200">
                                <div className="flex justify-between text-sm font-bold text-slate-700 mb-3">
                                    <span className="flex items-center gap-2">
                                        <span className="relative flex h-3 w-3">
                                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary-400 opacity-75"></span>
                                            <span className="relative inline-flex rounded-full h-3 w-3 bg-primary-500"></span>
                                        </span>
                                        Batch Processing ({batchData.stats.completed}/{batchData.stats.total} Completed)
                                    </span>
                                    <span className="font-mono text-slate-500">Running...</span>
                                </div>
                                <div className="w-full bg-slate-200 rounded-full h-3 border border-slate-300/50 overflow-hidden">
                                    <div
                                        className="h-full rounded-full transition-all duration-500 bg-gradient-to-r from-primary-500 to-secondary-500"
                                        style={{ width: `${(batchData.stats.completed / batchData.stats.total) * 100}%` }}
                                    >
                                        <div className="w-full h-full animate-pulse bg-white/30"></div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Split Layout: Left Data, Right Visualizer */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                        {/* LEFT COLUMN: Details & Table */}
                        <div className="space-y-6">

                            {/* Summary Card */}
                            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                                <h2 className="text-lg font-bold text-slate-900 mb-4">Batch Summary</h2>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
                                        <div className="text-sm text-slate-500 mb-1">Total Ligands</div>
                                        <div className="text-2xl font-bold text-slate-900">{batchData.stats.total}</div>
                                    </div>
                                    <div className="p-4 bg-green-50 rounded-xl border border-green-100">
                                        <div className="text-sm text-green-600 mb-1">Success Rate</div>
                                        <div className="text-2xl font-bold text-green-700">{batchData.stats.success_rate.toFixed(0)}%</div>
                                    </div>
                                    <div className="col-span-2 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-100">
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <div className="text-sm text-blue-600 font-bold uppercase mb-1">Top Hit Affinity</div>
                                                <div className="text-3xl font-bold text-blue-800">
                                                    {batchData.stats.best_affinity ? batchData.stats.best_affinity.toFixed(2) : '-'} <span className="text-sm font-normal text-blue-600">kcal/mol</span>
                                                </div>
                                            </div>
                                            <div className="text-4xl">🏆</div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Detailed Results Table */}
                            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                                <div className="px-6 py-4 border-b border-slate-200 bg-slate-50 flex justify-between items-center">
                                    <h3 className="font-bold text-slate-700">Ligand Rankings</h3>
                                    <span className="text-xs text-slate-500">{sortedJobs.length} results</span>
                                </div>
                                <div className="overflow-x-auto max-h-[600px]">
                                    <table className="min-w-full divide-y divide-slate-200">
                                        <thead className="bg-slate-50 sticky top-0 z-10 shadow-sm">
                                            <tr>
                                                <th onClick={() => handleSort('ligand_filename')} className="px-6 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider cursor-pointer hover:bg-slate-100">
                                                    Ligand {sortConfig.key === 'ligand_filename' && (sortConfig.direction === 'ascending' ? '↑' : '↓')}
                                                </th>
                                                <th onClick={() => handleSort('binding_affinity')} className="px-6 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider cursor-pointer hover:bg-slate-100">
                                                    Affinity {sortConfig.key === 'binding_affinity' && (sortConfig.direction === 'ascending' ? '↑' : '↓')}
                                                </th>
                                                <th className="px-6 py-3 text-right"></th>
                                            </tr>
                                        </thead>
                                        <tbody className="bg-white divide-y divide-slate-200">
                                            {sortedJobs.map((job) => (
                                                <tr key={job.id} className="hover:bg-slate-50 transition-colors">
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">
                                                        {job.ligand_filename.replace('.pdbqt', '')}
                                                        {job.id === firstJobId && <span className="ml-2 text-xs bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded">Viewing</span>}
                                                    </td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700 font-bold">
                                                        {job.binding_affinity ? job.binding_affinity.toFixed(1) : '-'}
                                                    </td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                                                        <Link to={`/dock/${job.id}`} className="text-primary-600 hover:text-primary-800 font-medium">
                                                            View &rarr;
                                                        </Link>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>

                        </div>

                        {/* RIGHT COLUMN: Sticky 3D Viewer */}
                        <div className="relative">
                            <div className="sticky top-24">
                                <div className="bg-white rounded-2xl shadow-lg border border-slate-200 overflow-hidden">
                                    <div className="p-4 bg-slate-50 border-b border-slate-200 flex justify-between items-center">
                                        <h3 className="font-bold text-slate-900 flex items-center gap-2">
                                            <span className="text-xl">🌟</span> Top Rank Visualization
                                        </h3>
                                        {firstJobId && (
                                            <Link to={`/dock/${firstJobId}`} className="text-xs bg-white border border-slate-300 px-2 py-1 rounded hover:bg-slate-50">
                                                Full Analysis
                                            </Link>
                                        )}
                                    </div>
                                    <div className="h-[600px] w-full relative bg-slate-900">
                                        {firstJobPdbqt ? (
                                            <MoleculeViewer
                                                pdbqtData={firstJobPdbqt}
                                                receptorData={firstJobReceptor}
                                                width="100%"
                                                height="100%"
                                                title="Best Binder"
                                            />
                                        ) : (
                                            <div className="flex flex-col items-center justify-center h-full text-slate-400 p-8 text-center">
                                                <div className="text-4xl mb-4">🧬</div>
                                                <p>Select a job or wait for results to view structure.</p>
                                                {loading && <p className="text-sm mt-2 animate-pulse">Loading data...</p>}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </main>
        </div>
    )
}
