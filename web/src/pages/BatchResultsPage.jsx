import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'
import MoleculeViewer from '../components/MoleculeViewer'
import AdmetRadar from '../components/AdmetRadar' // [NEW] Import Radar
import { trackEvent } from '../services/analytics' // Import Analytics
import { ChevronLeft, Download, Eye, Maximize2, RefreshCw, BarChart2, Star, Zap, Activity, ShieldCheck, AlertTriangle, ThumbsUp, ThumbsDown, FileCode } from 'lucide-react'

export default function BatchResultsPage() {
    const { batchId } = useParams()
    const [batchData, setBatchData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [sortConfig, setSortConfig] = useState({ key: 'binding_affinity', direction: 'ascending' })

    // Workbench State
    const [activeTab, setActiveTab] = useState('structure') // 'structure' | 'admet'

    // Viewer State
    const [firstJobPdbqt, setFirstJobPdbqt] = useState(null)
    const [firstJobReceptor, setFirstJobReceptor] = useState(null)
    const [firstJobId, setFirstJobId] = useState(null)
    const [firstJobName, setFirstJobName] = useState('')

    // ADMET State
    const [admetData, setAdmetData] = useState(null)
    const [admetLoading, setAdmetLoading] = useState(false)
    const [isSplitView, setIsSplitView] = useState(false)

    // Auto-Refresh
    useEffect(() => {
        if (batchId) fetchBatchDetails(batchId)
    }, [batchId])

    // [NEW] Real-time Updates via Supabase
    useEffect(() => {
        if (!batchId) return

        const channel = supabase
            .channel(`realtime:batch:${batchId}`)
            .on(
                'postgres_changes',
                {
                    event: '*', // Listen for inserts/updates
                    schema: 'public',
                    table: 'jobs',
                    filter: `batch_id=eq.${batchId}`
                },
                () => {
                    // Refresh data instantly on any change
                    fetchBatchDetails(batchId, true)
                }
            )
            .subscribe()

        return () => {
            supabase.removeChannel(channel)
        }
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

            // Auto-select best hit if not selected yet
            if (data.jobs && data.jobs.length > 0 && !firstJobId) {
                const validJobs = data.jobs.filter(j => j.status === 'SUCCEEDED' && j.binding_affinity !== null)
                if (validJobs.length > 0) {
                    const bestJob = validJobs.sort((a, b) => a.binding_affinity - b.binding_affinity)[0]
                    handleJobSelect(bestJob, session.access_token) // Use unified handler
                }
            }

        } catch (err) {
            console.error(err)
            if (!background) setError(err.message)
        } finally {
            if (!background) setLoading(false)
        }
    }

    const fetchJobStructure = async (jobId, token) => {
        try {
            setFirstJobPdbqt(null) // Reset to show loading
            const res = await fetch(`${API_URL}/jobs/${jobId}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            })
            if (!res.ok) return
            const jobData = await res.json()

            if (jobData.download_urls?.output_vina || jobData.download_urls?.output) {
                const url = jobData.download_urls.output_vina || jobData.download_urls.output
                const pdbqtRes = await fetch(url)
                setFirstJobPdbqt(await pdbqtRes.text())
            }
            if (jobData.download_urls?.receptor) {
                const recRes = await fetch(jobData.download_urls.receptor)
                setFirstJobReceptor(await recRes.text())
            }
        } catch (e) {
            console.error("Failed to load 3D structure", e)
        }
    }

    const fetchJobAdmet = async (jobId, token) => {
        try {
            setAdmetLoading(true)
            setAdmetData(null)
            const res = await fetch(`${API_URL}/jobs/${jobId}/admet`, {
                headers: { 'Authorization': `Bearer ${token}` }
            })
            if (!res.ok) throw new Error("ADMET fetch failed")
            const data = await res.json()
            setAdmetData(data)
        } catch (e) {
            console.error("Failed to load ADMET data", e)
        } finally {
            setAdmetLoading(false)
        }
    }

    const handleJobSelect = async (job, tokenOverride = null) => {
        if (job.status !== 'SUCCEEDED') return

        let token = tokenOverride
        if (!token) {
            const { data: { session } } = await supabase.auth.getSession()
            token = session?.access_token
        }
        if (!token) return

        setFirstJobId(job.id)
        setFirstJobName(job.ligand_filename)

        // Parallel Fetch or Lazy based on Tab? 
        // Fetch structure always as it's the primary view
        fetchJobStructure(job.id, token)

        // If on ADMET tab, fetch ADMET immediately
        if (activeTab === 'admet') {
            fetchJobAdmet(job.id, token)
        } else {
            // Clear old ADMET data so if they switch tabs it re-fetches relevant data
            setAdmetData(null)
        }
    }

    // Effect to fetch ADMET when switching TO the tab if missing
    useEffect(() => {
        if (activeTab === 'admet' && firstJobId && !admetData && !admetLoading) {
            supabase.auth.getSession().then(({ data: { session } }) => {
                if (session) fetchJobAdmet(firstJobId, session.access_token)
            })
        }
    }, [activeTab, firstJobId])

    // [NEW] Universal Download Handler
    const handleDownload = async (e, job, type) => {
        e.stopPropagation() // Prevent row selection
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) return

            const res = await fetch(`${API_URL}/jobs/${job.id}/files/${type}`, {
                headers: { 'Authorization': `Bearer ${session.access_token}` }
            })

            if (!res.ok) throw new Error('Failed to get download URL')

            const data = await res.json()
            if (data.url) {
                // Open in new tab (presigned S3 URL)
                window.open(data.url, '_blank')
            }
        } catch (err) {
            console.error(err)
            alert('Failed to download file. It might not exist yet.')
        }
    }

    const handleFeedback = async (e, job, rating) => {
        e.stopPropagation() // Prevent row selection

        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) return

            // Optimistic Update (Optional: could add local state to track processed IDs)
            // But ideally we'd want to reload data or update local cache
            // For MVP, just visual feedback via toast or simple console log, or update batchData locally

            const res = await fetch(`${API_URL}/feedback/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session.access_token}`
                },
                body: JSON.stringify({
                    job_id: job.id,
                    rating: rating
                })
            })

            if (!res.ok) throw new Error('Feedback failed')

            // Update local state to show the feedback (simplified)
            // Ideally we'd fetch this from DB, but let's just alert
            alert(`Draft: Feedback ${rating === 1 ? 'Positive' : 'Negative'} Recorded!`)

        } catch (err) {
            console.error(err)
            alert('Failed to submit feedback')
        }
    }


    const handleSort = (key) => {
        let direction = 'ascending'
        if (sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending'
        }
        setSortConfig({ key, direction })
    }

    // Helper to get affinity from various potential fields
    const getAffinity = (job) => {
        return job.binding_affinity ?? job.vina_score ?? job.docking_score ?? null
    }

    const sortedJobs = batchData?.jobs ? [...batchData.jobs].filter(j => j).sort((a, b) => {
        const valA = sortConfig.key === 'binding_affinity' ? getAffinity(a) : a[sortConfig.key]
        const valB = sortConfig.key === 'binding_affinity' ? getAffinity(b) : b[sortConfig.key]

        if (valA === null) return 1
        if (valB === null) return -1
        if (valA < valB) return sortConfig.direction === 'ascending' ? -1 : 1
        if (valA > valB) return sortConfig.direction === 'ascending' ? 1 : -1
        return 0
    }) : []

    const getAffinityColor = (score) => {
        if (!score) return 'text-slate-400'
        if (score < -9.0) return 'text-emerald-600 font-bold bg-emerald-50 px-2 py-0.5 rounded border border-emerald-100'
        if (score < -7.0) return 'text-blue-600 font-medium'
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
                <span className="text-5xl animate-spin text-indigo-600">🔄</span>
                <p className="text-slate-500 font-medium">Retrieving results...</p>
            </div>
        </div>
    )

    if (error || !batchData) return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50">
            <div className="text-center max-w-md p-8 bg-white rounded-2xl shadow-xl">
                <div className="text-4xl mb-4">⚠️</div>
                <h2 className="text-xl font-bold text-slate-900">Batch Not Found</h2>
                <Link to="/dashboard" className="btn-secondary mt-6 inline-flex">Return to Dashboard</Link>
            </div>
        </div>
    )

    return (
        <div className="h-screen flex flex-col bg-slate-50 overflow-hidden">

            {/* 1. Header Bar */}
            <div className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6 flex-shrink-0 z-30">
                <div className="flex items-center gap-4">
                    <Link to="/dashboard" className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors">
                        <span className="text-xl">⬅️</span>
                    </Link>
                    <div>
                        <h1 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                            Batch Analysis <span className="px-2 py-0.5 rounded text-xs bg-slate-100 text-slate-500 font-mono">{batchId.slice(0, 8)}</span>
                        </h1>
                        <div className="text-xs text-slate-500 flex items-center gap-2">
                            {batchData.status === 'SUBMITTED' && (
                                <span className="flex items-center gap-1 text-amber-600 font-bold"><span className="animate-pulse">⏳</span> Queued (Runnable)</span>
                            )}
                            {batchData.status === 'RUNNING' && (
                                <span className="flex items-center gap-1 text-blue-600 font-bold"><span className="animate-spin">🔄</span> Processing ({batchData.stats?.completed || 0}/{batchData.stats?.total || 0})</span>
                            )}
                            {batchData.status === 'SUCCEEDED' && (
                                <span className="text-emerald-600 font-bold flex items-center gap-1"><span>⭐</span> Complete</span>
                            )}
                            {batchData.status === 'FAILED' && (
                                <span className="text-red-600 font-bold flex items-center gap-1"><span>⚠️</span> Failed</span>
                            )}
                            <span className="text-slate-300">|</span>
                            <span>{batchData.stats?.total || 0} Ligands</span>
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    <button onClick={downloadCSV} className="btn-secondary btn-sm flex items-center gap-2">
                        <span>📥</span> Export CSV
                    </button>
                    {/* PDF Report (Phase 9) */}
                    <button
                        onClick={async () => {
                            const { data: { session } } = await supabase.auth.getSession()
                            if (!session) return

                            // Track PDF Download
                            trackEvent('report:downloaded', {
                                batch_id: batchId,
                                format: 'pdf'
                            });

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
                        className="btn-secondary btn-sm flex items-center gap-2"
                    >
                        <span>📊</span> PDF Report
                    </button>
                </div>
            </div>

            {/* 2. Main Workbench Area */}
            <div className="flex-1 flex flex-col md:flex-row overflow-hidden">

                {/* LEFT: Data Table with Heatmap */}
                <div className="w-full h-1/2 md:h-full md:w-1/3 md:min-w-[400px] border-b md:border-b-0 md:border-r border-slate-200 bg-white flex flex-col">
                    <div className="p-4 border-b border-slate-200 bg-slate-50 flex justify-between items-center">
                        <h3 className="font-bold text-slate-700 text-sm uppercase tracking-wider">Results Table</h3>
                        <div className="text-xs text-slate-500">Sorted by Affinity</div>
                    </div>
                    <div className="flex-1 overflow-auto">
                        <table className="w-full text-sm text-left">
                            <thead className="text-xs text-slate-500 uppercase bg-slate-50 sticky top-0 z-10 shadow-sm">
                                <tr>
                                    <th className="px-4 py-3 text-xs w-[100px]">ID</th>
                                    <th className="px-4 py-3">Status</th>
                                    <th onClick={() => handleSort('ligand_filename')} className="px-4 py-3 cursor-pointer hover:bg-slate-100">Ligand</th>
                                    <th onClick={() => handleSort('vina_score')} className="px-4 py-3 cursor-pointer hover:bg-slate-100 text-right text-[10px]">Vina</th>
                                    <th onClick={() => handleSort('docking_score')} className="px-4 py-3 cursor-pointer hover:bg-slate-100 text-right text-[10px]">Gnina</th>
                                    <th onClick={() => handleSort('binding_affinity')} className="px-4 py-3 cursor-pointer hover:bg-slate-100 text-right text-[10px]">Consensus</th>
                                    <th className="px-4 py-3 text-center">Files</th>
                                    <th className="px-4 py-3 text-center">Feedback</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {sortedJobs.map((job) => (
                                    <tr
                                        key={job.id}
                                        onClick={() => handleJobSelect(job)}
                                        className={`cursor-pointer transition-colors hover:bg-indigo-50 ${firstJobId === job.id ? 'bg-indigo-50 border-l-4 border-indigo-500' : ''}`}
                                    >
                                        <td className="px-4 py-3 font-mono text-xs text-slate-400">
                                            {job.id.slice(0, 8)}
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${job.status === 'SUCCEEDED' ? 'bg-emerald-50 text-emerald-600 border-emerald-100' :
                                                job.status === 'FAILED' ? 'bg-red-50 text-red-600 border-red-100' :
                                                    'bg-blue-50 text-blue-600 border-blue-100'
                                                }`}>
                                                {job.status}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3 font-medium text-slate-900 truncate max-w-[150px]" title={job.ligand_filename}>
                                            {(job.ligand_filename || 'Unknown').replace('.pdbqt', '')}
                                        </td>
                                        <td className="px-4 py-3 text-right font-mono text-xs">
                                            {job.vina_score && !isNaN(Number(job.vina_score)) ? <span className="text-blue-600">{Number(job.vina_score).toFixed(2)}</span> : <span className="text-slate-300">-</span>}
                                        </td>
                                        <td className="px-4 py-3 text-right font-mono text-xs">
                                            {job.docking_score && !isNaN(Number(job.docking_score)) ? <span className="text-purple-600">{Number(job.docking_score).toFixed(2)}</span> : <span className="text-slate-300">-</span>}
                                        </td>
                                        <td className="px-4 py-3 text-right font-mono font-bold">
                                            {(() => {
                                                const rawAff = getAffinity(job);
                                                const aff = rawAff !== null ? Number(rawAff) : null;

                                                if (aff !== null && !isNaN(aff) && aff !== 0) {
                                                    return <span className={getAffinityColor(aff)}>{aff.toFixed(2)}</span>
                                                }
                                                if (job.status === 'SUCCEEDED') return <span className="text-red-500 text-xs">Error</span>
                                                return <span className="text-slate-300">-</span>
                                            })()}
                                        </td>
                                        <td className="px-4 py-3 text-center flex justify-center gap-2">
                                            <button
                                                onClick={(e) => handleDownload(e, job, 'output')}
                                                className="p-1 hover:bg-indigo-100 text-slate-400 hover:text-indigo-600 rounded"
                                                title="Download Docked PDBQT"
                                            >
                                                <Download size={14} />
                                            </button>
                                            <button
                                                onClick={(e) => handleDownload(e, job, 'log')}
                                                className="p-1 hover:bg-slate-200 text-slate-400 hover:text-slate-600 rounded"
                                                title="View Log"
                                            >
                                                <FileCode size={14} />
                                            </button>
                                            <button
                                                onClick={(e) => handleDownload(e, job, 'config')}
                                                className="p-1 hover:bg-slate-200 text-slate-400 hover:text-slate-600 rounded"
                                                title="View Config"
                                            >
                                                <Zap size={14} />
                                            </button>
                                        </td>
                                        <td className="px-4 py-3 text-center flex justify-center gap-2">
                                            <button
                                                onClick={(e) => handleFeedback(e, job, 1)}
                                                className="p-1 hover:bg-green-100 text-slate-400 hover:text-green-600 rounded transition-colors"
                                                title="Good Result"
                                            >
                                                <span>👍</span>
                                            </button>
                                            <button
                                                onClick={(e) => handleFeedback(e, job, -1)}
                                                className="p-1 hover:bg-red-100 text-slate-400 hover:text-red-600 rounded transition-colors"
                                                title="Bad Result"
                                            >
                                                <span>👎</span>
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* RIGHT: Visualization Panel (Simple 3D + Analysis Action) */}
                <div className="flex-1 bg-slate-100 relative flex flex-col">
                    {/* Simple Header for Panel */}
                    <div className="absolute top-4 left-4 z-10 pointer-events-none">
                        <div className="bg-white/90 backdrop-blur shadow-sm rounded-lg px-3 py-1.5 border border-slate-200">
                            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1">
                                <span>📈</span> 3D Preview
                            </h3>
                        </div>
                    </div>

                    {/* Action Button for Deep Analysis */}
                    {firstJobSelected && (
                        <div className="absolute top-4 right-4 z-10">
                            <button
                                onClick={() => navigate(`/jobs/${firstJobId}/analysis`)}
                                className="bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg shadow-indigo-200 rounded-xl px-4 py-2 text-sm font-bold flex items-center gap-2 transition-all transform hover:scale-105"
                            >
                                <Zap size={16} /> Deep Analysis
                            </button>
                        </div>
                    )}

                    {/* Content Container */}
                    <div className="flex-1 w-full h-full relative mt-0">
                        {/* VIEW: Simple 3D Structure */}
                        <div className="w-full h-full opacity-100 z-0">
                            {firstJobPdbqt ? (
                                <MoleculeViewer
                                    pdbqtData={firstJobPdbqt}
                                    receptorData={firstJobReceptor}
                                    width="100%"
                                    height="100%"
                                    title=""
                                />
                            ) : (
                                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400">
                                    <span className="text-6xl mb-4 opacity-50">⚡</span>
                                    <p className="text-lg font-medium">Select a ligand to preview</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

            </div>
        </div>
    )
}
