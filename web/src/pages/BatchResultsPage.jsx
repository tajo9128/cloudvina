import { useState, useEffect } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'
import { trackEvent } from '../services/analytics'
import { Download, FileCode, Zap, ArrowRight, LayoutGrid, List, Activity, CheckCircle2, XCircle, Clock, Filter, Search } from 'lucide-react'

export default function BatchResultsPage() {
    const { batchId } = useParams()
    const navigate = useNavigate()
    const [batchData, setBatchData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [sortConfig, setSortConfig] = useState({ key: 'binding_affinity', direction: 'ascending' })
    const [currentPage, setCurrentPage] = useState(1)
    const itemsPerPage = 10

    // Auto-Refresh (Polling & Realtime)
    useEffect(() => {
        if (batchId) {
            fetchBatchDetails(batchId)
            // Polling every 10s as backup/primary sync (triggers lazy repair)
            const interval = setInterval(() => fetchBatchDetails(batchId, true), 10000)
            return () => clearInterval(interval)
        }
    }, [batchId])

    // Real-time Updates (Supabase Subscription)
    useEffect(() => {
        if (!batchId) return
        const channel = supabase
            .channel(`realtime:batch:${batchId}`)
            .on('postgres_changes', { event: '*', schema: 'public', table: 'jobs', filter: `batch_id=eq.${batchId}` }, () => fetchBatchDetails(batchId, true))
            .subscribe()
        return () => { supabase.removeChannel(channel) }
    }, [batchId])

    const fetchBatchDetails = async (id, background = false) => {
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) return

            const response = await fetch(`${API_URL}/jobs/batch/${id}`, {
                headers: { 'Authorization': `Bearer ${session.access_token}` }
            })

            if (!response.ok) throw new Error('Failed to fetch batch details')

            const data = await response.json()
            setBatchData(data)

        } catch (err) {
            console.error(err)
            if (!background) setError(err.message)
        } finally {
            if (!background) setLoading(false)
        }
    }

    const handleDownload = async (e, job, type) => {
        e.stopPropagation()
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) return
            const res = await fetch(`${API_URL}/jobs/${job.id}/files/${type}`, {
                headers: { 'Authorization': `Bearer ${session.access_token}` }
            })
            if (!res.ok) throw new Error('Failed to get download URL')
            const data = await res.json()
            if (data.url) window.open(data.url, '_blank')
        } catch (err) {
            alert('Failed to download file. It might not exist yet.')
        }
    }

    const handleSort = (key) => {
        let direction = 'ascending'
        if (sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending'
        }
        setSortConfig({ key, direction })
    }

    const getAffinity = (job) => job.binding_affinity ?? job.vina_score ?? job.docking_score ?? null

    const sortedJobs = batchData?.jobs ? [...batchData.jobs].filter(j => j).sort((a, b) => {
        const valA = sortConfig.key === 'binding_affinity' ? getAffinity(a) : a[sortConfig.key]
        const valB = sortConfig.key === 'binding_affinity' ? getAffinity(b) : b[sortConfig.key]
        if (valA === null) return 1
        if (valB === null) return -1
        if (valA < valB) return sortConfig.direction === 'ascending' ? -1 : 1
        if (valA > valB) return sortConfig.direction === 'ascending' ? 1 : -1
        return 0
    }) : []

    const totalPages = Math.ceil(sortedJobs.length / itemsPerPage)
    const paginatedJobs = sortedJobs.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage)

    const getAffinityColor = (score) => {
        if (!score) return 'text-slate-400'
        if (score < -9.0) return 'text-emerald-600 font-bold bg-emerald-50 px-2 py-0.5 rounded border border-emerald-100'
        if (score < -7.0) return 'text-indigo-600 font-medium'
        return 'text-slate-600'
    }

    const downloadCSV = () => {
        if (!batchData?.jobs) return
        const headers = ['Ligand Name', 'Status', 'Binding Affinity (kcal/mol)', 'Job ID']
        const rows = batchData.jobs.map(job => [
            (job.ligand_filename || 'unknown.pdbqt').replace('.pdbqt', ''),
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

    if (loading) return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50">
            <div className="flex flex-col items-center gap-4">
                <div className="relative">
                    <div className="w-16 h-16 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin"></div>
                    <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-indigo-600">AI</div>
                </div>
                <p className="text-slate-500 font-medium animate-pulse">Retrieving batch data...</p>
            </div>
        </div>
    )

    if (error || !batchData) return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50">
            <div className="text-center max-w-md p-8 bg-white rounded-3xl shadow-xl border border-slate-100">
                <div className="w-16 h-16 bg-red-50 text-red-500 rounded-full flex items-center justify-center mx-auto mb-4">
                    <XCircle size={32} />
                </div>
                <h2 className="text-xl font-bold text-slate-900 mb-2">Batch Not Found</h2>
                <Link to="/dashboard" className="btn-secondary mt-6 inline-flex">Return to Dashboard</Link>
            </div>
        </div>
    )

    // Calculate Stats
    const totalJobs = batchData.stats?.total || 0
    const successCount = batchData.jobs?.filter(j => j.status === 'SUCCEEDED').length || 0
    const runningCount = batchData.jobs?.filter(j => ['RUNNING', 'QUEUED', 'STARTING', 'submitted', 'SUBMITTED', 'RUNNABLE'].includes(j.status)).length || 0
    const failedCount = batchData.jobs?.filter(j => j.status === 'FAILED').length || 0

    // Progress Calculation
    const progressPercent = totalJobs > 0 ? Math.round(((successCount + failedCount) / totalJobs) * 100) : 0
    const estTimeRemaining = runningCount > 0 ? Math.ceil(runningCount * 2.5) : 0 // Approx 2.5 mins per parallel slot (heuristic)

    // Average Affinity (only successes)
    const validAffinities = batchData.jobs?.map(getAffinity).filter(a => a !== null && !isNaN(a)) || []
    const avgAffinity = validAffinities.length > 0
        ? (validAffinities.reduce((a, b) => a + b, 0) / validAffinities.length).toFixed(2)
        : '-'

    return (
        <div className="h-screen flex flex-col bg-slate-50 overflow-hidden font-sans">

            {/* Top Navigation Bar */}
            <div className="h-20 bg-white border-b border-slate-200 flex items-center justify-between px-8 flex-shrink-0 z-30 shadow-sm">
                <div className="flex items-center gap-6">
                    <Link to="/dashboard" className="w-10 h-10 flex items-center justify-center hover:bg-slate-100 rounded-full text-slate-400 transition-colors">
                        <ArrowRight className="rotate-180 w-5 h-5" />
                    </Link>
                    <div>
                        <div className="flex items-center gap-3">
                            <h1 className="text-xl font-bold text-slate-900 tracking-tight">Batch Results</h1>
                            <span className="px-2 py-0.5 bg-slate-100 border border-slate-200 rounded text-xs font-mono text-slate-500">
                                {batchId.slice(0, 8)}...
                            </span>
                        </div>
                        <div className="text-xs text-slate-500 mt-1 flex items-center gap-2">
                            Created {new Date(batchData.created_at || Date.now()).toLocaleDateString()}
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    <button onClick={downloadCSV} className="btn-secondary btn-sm flex items-center gap-2 border-slate-200">
                        <Download size={16} /> <span>Export CSV</span>
                    </button>
                    <button
                        onClick={async () => {
                            // ... existing PDF logic
                            const { data: { session } } = await supabase.auth.getSession()
                            if (!session) return
                            trackEvent('report:downloaded', { batch_id: batchId, format: 'pdf' });
                            const res = await fetch(`${API_URL}/jobs/batch/${batchId}/report-pdf`, {
                                headers: { 'Authorization': `Bearer ${session.access_token}` }
                            })
                            if (!res.ok) { alert('Failed to generate PDF'); return }
                            const blob = await res.blob()
                            const url = window.URL.createObjectURL(blob)
                            const a = document.createElement('a')
                            a.href = url
                            a.download = `BioDockify_Report_${batchId.slice(0, 8)}.pdf`
                            a.click()
                            window.URL.revokeObjectURL(url)
                        }}
                        className="btn-primary btn-sm flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white shadow-md shadow-indigo-100"
                    >
                        <span>📄</span> <span>Generate Report</span>
                    </button>
                </div>
            </div>

            {/* Dashboard Content */}
            <div className="flex-1 overflow-auto p-6 lg:p-10">
                <div className="max-w-7xl mx-auto space-y-8">

                    {/* Progress Bar Section */}
                    {progressPercent < 100 && (
                        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                            <div className="flex justify-between items-end mb-2">
                                <div>
                                    <h3 className="font-bold text-slate-900">Batch Progress</h3>
                                    <p className="text-sm text-slate-500 flex items-center gap-2">
                                        {runningCount > 0 ? (
                                            <>
                                                <span className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse"></span>
                                                Processing {runningCount} jobs... (~{estTimeRemaining} mins remaining)
                                            </>
                                        ) : "Batch initialization..."}
                                    </p>
                                </div>
                                <span className="text-2xl font-bold text-indigo-600">{progressPercent}%</span>
                            </div>
                            <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                                <div
                                    className="bg-indigo-600 h-full rounded-full transition-all duration-1000 ease-out"
                                    style={{ width: `${progressPercent}%` }}
                                >
                                    <div className="w-full h-full opacity-30 bg-[url('https://www.transparenttextures.com/patterns/diagonal-stripes.png')] animate-[slide_1s_linear_infinite]"></div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* 1. Metrics Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                        {/* Total Jobs */}
                        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-center gap-4">
                            <div className="p-3 bg-slate-100 rounded-xl text-slate-600">
                                <List size={24} />
                            </div>
                            <div>
                                <div className="text-sm font-bold text-slate-500 uppercase tracking-wide">Total Jobs</div>
                                <div className="text-3xl font-bold text-slate-900">{totalJobs}</div>
                            </div>
                        </div>

                        {/* Completed */}
                        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-center gap-4">
                            <div className="p-3 bg-emerald-50 rounded-xl text-emerald-600">
                                <CheckCircle2 size={24} />
                            </div>
                            <div>
                                <div className="text-sm font-bold text-slate-500 uppercase tracking-wide">Completed</div>
                                <div className="text-3xl font-bold text-slate-900">{successCount}</div>
                            </div>
                        </div>

                        {/* Best Affinity */}
                        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-center gap-4">
                            <div className="p-3 bg-indigo-50 rounded-xl text-indigo-600">
                                <Zap size={24} />
                            </div>
                            <div>
                                <div className="text-sm font-bold text-slate-500 uppercase tracking-wide">Best Affinity</div>
                                <div className="text-3xl font-bold text-indigo-600">
                                    {batchData.jobs?.reduce((min, job) => Math.min(min, getAffinity(job) || 0), 0)?.toFixed(1) || '-'}
                                </div>
                            </div>
                        </div>

                        {/* Avg Affinity */}
                        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-center gap-4">
                            <div className="p-3 bg-violet-50 rounded-xl text-violet-600">
                                <Activity size={24} />
                            </div>
                            <div>
                                <div className="text-sm font-bold text-slate-500 uppercase tracking-wide">Avg. Score</div>
                                <div className="text-3xl font-bold text-slate-900">{avgAffinity}</div>
                            </div>
                        </div>
                    </div>

                    {/* 2. Main Results Table */}
                    <div className="bg-white rounded-3xl shadow-lg border border-slate-200 overflow-hidden">
                        {/* Table Header / Toolbar */}
                        <div className="p-6 border-b border-slate-100 flex items-center justify-between bg-slate-50/50 backdrop-blur-sm">
                            <div className="flex items-center gap-3">
                                <h2 className="text-lg font-bold text-slate-800">Job Results</h2>
                                <span className="px-2 py-0.5 bg-slate-200 text-slate-600 rounded-full text-xs font-bold">{sortedJobs.length} Items</span>
                            </div>
                            <div className="flex items-center gap-3">
                                {/* Search Placeholder - Functional in future */}
                                <div className="relative group">
                                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 group-hover:text-indigo-500 transition-colors" size={16} />
                                    <input
                                        type="text"
                                        placeholder="Search ligands..."
                                        className="pl-10 pr-4 py-2 rounded-xl bg-white border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all w-64"
                                    />
                                </div>
                                <button className="p-2 bg-white border border-slate-200 rounded-xl text-slate-500 hover:text-indigo-600 hover:border-indigo-200 transition-all">
                                    <Filter size={18} />
                                </button>
                            </div>
                        </div>

                        <div className="overflow-x-auto">
                            <table className="w-full text-left border-collapse">
                                <thead>
                                    <tr className="bg-slate-50 border-b border-slate-200 text-xs text-slate-500 uppercase tracking-wider font-semibold">
                                        <th className="px-8 py-5">Job ID</th>
                                        <th className="px-6 py-5">Sim Status</th>
                                        <th onClick={() => handleSort('ligand_filename')} className="px-6 py-5 cursor-pointer hover:text-indigo-600 transition-colors">Ligand Name</th>
                                        <th onClick={() => handleSort('vina_score')} className="px-6 py-5 text-right cursor-pointer hover:text-indigo-600 transition-colors">Vina (kcal/mol)</th>
                                        <th onClick={() => handleSort('docking_score')} className="px-6 py-5 text-right cursor-pointer hover:text-indigo-600 transition-colors">Gnina (kcal/mol)</th>
                                        <th onClick={() => handleSort('binding_affinity')} className="px-6 py-5 text-right cursor-pointer hover:text-indigo-600 transition-colors">Consensus</th>
                                        <th className="px-6 py-5 text-center">Quick Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {paginatedJobs.map((job) => (
                                        <tr
                                            key={job.id}
                                            onClick={() => navigate(`/dock/${job.id}`)}
                                            className="group hover:bg-indigo-50/50 cursor-pointer transition-colors duration-150"
                                        >
                                            <td className="px-8 py-5 font-mono text-xs text-slate-400 group-hover:text-indigo-500">
                                                {job.id.slice(0, 8)}
                                            </td>
                                            <td className="px-6 py-5">
                                                <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold border capitalize tracking-wide ${job.status === 'SUCCEEDED' ? 'bg-emerald-50 text-emerald-600 border-emerald-100' :
                                                    job.status === 'FAILED' ? 'bg-red-50 text-red-600 border-red-100' :
                                                        'bg-amber-50 text-amber-600 border-amber-100 animate-pulse'
                                                    }`}>
                                                    <span className={`w-1.5 h-1.5 rounded-full ${job.status === 'SUCCEEDED' ? 'bg-emerald-500' :
                                                        job.status === 'FAILED' ? 'bg-red-500' : 'bg-amber-500'
                                                        }`}></span>
                                                    {job.status.toLowerCase()}
                                                </span>
                                            </td>
                                            <td className="px-6 py-5">
                                                <div className="font-bold text-slate-900 group-hover:text-indigo-700 transition-colors">
                                                    {(job.ligand_filename || 'Unknown').replace('.pdbqt', '')}
                                                </div>
                                            </td>
                                            <td className="px-6 py-5 text-right font-mono text-sm text-slate-500">
                                                {job.vina_score && !isNaN(Number(job.vina_score)) ? Number(job.vina_score).toFixed(2) : '-'}
                                            </td>
                                            <td className="px-6 py-5 text-right font-mono text-sm text-slate-500">
                                                {job.docking_score && !isNaN(Number(job.docking_score)) ? Number(job.docking_score).toFixed(2) : '-'}
                                            </td>
                                            <td className="px-6 py-5 text-right font-mono font-bold text-sm">
                                                {(() => {
                                                    const rawAff = getAffinity(job);
                                                    const aff = rawAff !== null ? Number(rawAff) : null;
                                                    if (aff !== null && !isNaN(aff) && aff !== 0) return <span className={getAffinityColor(aff)}>{aff.toFixed(2)}</span>
                                                    return <span className="text-slate-300">-</span>
                                                })()}
                                            </td>
                                            <td className="px-6 py-5 text-center">
                                                <div className="flex justify-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity items-center">
                                                    {job.status === 'SUCCEEDED' && (
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                navigate(`/dock/${job.id}`)
                                                            }}
                                                            className="px-3 py-1.5 bg-indigo-50 border border-indigo-100 rounded-lg text-indigo-600 text-xs font-bold hover:bg-indigo-100 transition-all flex items-center gap-1"
                                                        >
                                                            <Zap size={12} /> Analyze
                                                        </button>
                                                    )}
                                                    <button onClick={(e) => handleDownload(e, job, 'output')} className="p-2 bg-white border border-slate-200 rounded-lg text-slate-500 hover:text-indigo-600 hover:border-indigo-200 transition-all shadow-sm" title="Download structure">
                                                        <Download size={14} />
                                                    </button>
                                                    <button onClick={(e) => handleDownload(e, job, 'log')} className="p-2 bg-white border border-slate-200 rounded-lg text-slate-500 hover:text-slate-800 hover:border-slate-300 transition-all shadow-sm" title="View Logs">
                                                        <FileCode size={14} />
                                                    </button>
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {/* Pagination Footer */}
                        <div className="p-6 border-t border-slate-200 flex items-center justify-between bg-slate-50">
                            <span className="text-sm font-medium text-slate-500">
                                Page {currentPage} of {totalPages || 1}
                            </span>
                            <div className="flex gap-2">
                                <button
                                    disabled={currentPage === 1}
                                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                    className="px-4 py-2 bg-white border border-slate-200 rounded-xl text-sm font-bold text-slate-600 disabled:opacity-50 hover:bg-slate-50 hover:border-slate-300 transition-all"
                                >
                                    Previous
                                </button>
                                <button
                                    disabled={currentPage === totalPages}
                                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                                    className="px-4 py-2 bg-white border border-slate-200 rounded-xl text-sm font-bold text-slate-600 disabled:opacity-50 hover:bg-slate-50 hover:border-slate-300 transition-all"
                                >
                                    Next
                                </button>
                            </div>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    )
}
