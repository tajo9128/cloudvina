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

    const sortedJobs = batchData?.jobs ? [...batchData.jobs].sort((a, b) => {
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
                            {['RUNNING', 'SUBMITTED'].includes(batchData.status) ? (
                                <span className="flex items-center gap-1 text-blue-600 font-bold"><span className="animate-spin">🔄</span> Processing ({batchData.stats.completed}/{batchData.stats.total})</span>
                            ) : (
                                <span className="text-emerald-600 font-bold flex items-center gap-1"><span>⭐</span> Complete</span>
                            )}
                            <span className="text-slate-300">|</span>
                            <span>{batchData.stats.total} Ligands</span>
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
                                    <th onClick={() => handleSort('ligand_filename')} className="px-4 py-3 cursor-pointer hover:bg-slate-100">Ligand</th>
                                    <th onClick={() => handleSort('binding_affinity')} className="px-4 py-3 cursor-pointer hover:bg-slate-100 text-right">Affinity</th>
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
                                        <td className="px-4 py-3 font-medium text-slate-900 truncate max-w-[150px]">
                                            {job.ligand_filename.replace('.pdbqt', '')}
                                            {job.status !== 'SUCCEEDED' && <span className="ml-2 text-[10px] bg-slate-100 px-1 rounded text-slate-500">{job.status}</span>}
                                        </td>
                                        <td className="px-4 py-3 text-right font-mono font-bold">
                                            <span className={getAffinityColor(getAffinity(job))}>
                                                {getAffinity(job) !== null ? getAffinity(job).toFixed(1) : '-'}
                                            </span>
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

                {/* RIGHT: Visualization Panel (3D + ADMET) */}
                <div className="flex-1 bg-slate-100 relative flex flex-col">
                    {/* Tab Bar */}
                    <div className="absolute top-4 left-4 right-4 z-10 flex justify-between items-start pointer-events-none">
                        <div className="bg-white/90 backdrop-blur shadow-lg rounded-xl p-2 border border-slate-200 pointer-events-auto flex gap-1">
                            <button
                                onClick={() => setActiveTab('structure')}
                                className={`px-4 py-2 rounded-lg text-sm font-bold flex items-center gap-2 transition-colors ${activeTab === 'structure' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-slate-50 text-slate-500'}`}
                            >
                                <span>📈</span> 3D Structure
                            </button>
                            <button
                                onClick={() => setActiveTab('admet')}
                                className={`px-4 py-2 rounded-lg text-sm font-bold flex items-center gap-2 transition-colors ${activeTab === 'admet' ? 'bg-purple-100 text-purple-700' : 'hover:bg-slate-50 text-slate-500'}`}
                            >
                                <span>🛡️</span> ADMET Profile <span className="text-[10px] px-1.5 bg-purple-200 rounded-full">NEW</span>
                            </button>
                        </div>

                        {/* Only show Viewer Controls if in Structure Mode */}
                        {/* Viewer Controls moved inside MoleculeViewer component */}
                    </div>

                    {/* Content Container */}
                    <div className="flex-1 w-full h-full relative mt-0">
                        {/* VIEW 1: 3D Structure */}
                        <div className={`w-full h-full transition-opacity duration-300 ${activeTab === 'structure' ? 'opacity-100 z-0' : 'opacity-0 z-[-1] absolute inset-0'}`}>
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
                                    <p className="text-lg font-medium">Select a ligand from the table to visualize</p>
                                </div>
                            )}
                        </div>

                        {/* VIEW 2: ADMET Analysis */}
                        {activeTab === 'admet' && (
                            <div className="w-full h-full bg-slate-50 pt-24 pb-8 px-8 overflow-y-auto">
                                <div className="max-w-4xl mx-auto">
                                    <div className="text-center mb-8">
                                        <h2 className="text-2xl font-bold text-slate-800">{firstJobName ? firstJobName.replace('.pdbqt', '') : 'Compound'} Analysis</h2>
                                        <p className="text-slate-500">Predicted Pharmacokinetics & Toxicity Profile</p>
                                    </div>

                                    {admetLoading ? (
                                        <div className="flex flex-col items-center justify-center py-20">
                                            <span className="text-5xl animate-spin mb-4">🔄</span>
                                            <p className="text-slate-400 text-lg">Running ADMET models...</p>
                                        </div>
                                    ) : admetData ? (
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                            {/* Left: Radar Chart */}
                                            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col items-center">
                                                <h3 className="font-bold text-slate-600 mb-6 uppercase tracking-wider text-sm">Molecular Properties</h3>
                                                <AdmetRadar data={admetData} width={340} height={340} />

                                                <div className="mt-8 grid grid-cols-2 gap-4 w-full">
                                                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 text-center">
                                                        <div className="text-xs text-slate-500 uppercase">Drug-Likeness</div>
                                                        <div className={`text-xl font-bold ${admetData.score >= 80 ? 'text-green-600' : 'text-amber-500'}`}>
                                                            {admetData.score}/100
                                                        </div>
                                                    </div>
                                                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 text-center">
                                                        <div className="text-xs text-slate-500 uppercase">Violations</div>
                                                        <div className="text-xl font-bold text-slate-700">{admetData.lipinski.violations}</div>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Right: Toxicity & Badges */}
                                            <div className="space-y-6">

                                                {/* AI Explanation Card */}
                                                {batchData?.jobs?.find(j => j.id === firstJobId)?.ai_explanation && (
                                                    <div className="bg-gradient-to-br from-indigo-50 to-white p-6 rounded-2xl shadow-sm border border-indigo-100 relative overflow-hidden">
                                                        <div className="absolute top-0 right-0 p-4 opacity-10">
                                                            <span className="text-8xl">⚡</span>
                                                        </div>
                                                        <h3 className="font-bold text-indigo-700 mb-3 uppercase tracking-wider text-sm flex items-center gap-2">
                                                            <span>⚡</span> AI Ranking Explanation
                                                        </h3>
                                                        <p className="text-slate-800 font-medium leading-relaxed relative z-10">
                                                            {batchData.jobs.find(j => j.id === firstJobId).ai_explanation}
                                                        </p>
                                                    </div>
                                                )}

                                                {/* Alerts Card */}
                                                <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                                                    <h3 className="font-bold text-slate-600 mb-4 uppercase tracking-wider text-sm">Toxicity Alerts</h3>
                                                    <div className="space-y-3">
                                                        {/* hERG */}
                                                        <div className="flex items-center justify-between p-3 rounded-lg border border-slate-100 bg-slate-50">
                                                            <div className="flex items-center gap-3">
                                                                <div className="p-2 bg-white rounded shadow-sm">❤️</div>
                                                                <div>
                                                                    <div className="font-bold text-slate-700 text-sm">hERG Liability</div>
                                                                    <div className="text-xs text-slate-400">Cardiotoxicity Risk</div>
                                                                </div>
                                                            </div>
                                                            <span className={`px-3 py-1 rounded-full text-xs font-bold ${admetData.herg.risk_level === 'Low' ? 'bg-green-100 text-green-700' :
                                                                admetData.herg.risk_level === 'Moderate' ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'
                                                                }`}>
                                                                {admetData.herg.risk_level}
                                                            </span>
                                                        </div>

                                                        {/* AMES */}
                                                        <div className="flex items-center justify-between p-3 rounded-lg border border-slate-100 bg-slate-50">
                                                            <div className="flex items-center gap-3">
                                                                <div className="p-2 bg-white rounded shadow-sm">🧬</div>
                                                                <div>
                                                                    <div className="font-bold text-slate-700 text-sm">AMES Mutagenicity</div>
                                                                    <div className="text-xs text-slate-400">Genotoxicity Risk</div>
                                                                </div>
                                                            </div>
                                                            <span className={`px-3 py-1 rounded-full text-xs font-bold ${admetData.ames.prediction === 'Negative' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                                                                }`}>
                                                                {admetData.ames.prediction}
                                                            </span>
                                                        </div>

                                                        {/* PAINS */}
                                                        {admetData.pains.passed ? (
                                                            <div className="flex items-center gap-2 text-xs text-green-600 font-medium px-2">
                                                                <span>🛡️</span> No PAINS alerts detected
                                                            </div>
                                                        ) : (
                                                            <div className="flex items-center gap-2 text-xs text-red-600 font-medium px-2 bg-red-50 p-2 rounded">
                                                                <span>⚠️</span> PAINS Alert: {admetData.pains.alerts.join(", ")}
                                                            </div>
                                                        )}

                                                        {/* CYP/DDI Risk (New) */}
                                                        {admetData.cyp && (
                                                            <div className="mt-4 pt-4 border-t border-slate-100">
                                                                <div className="flex items-center justify-between mb-2">
                                                                    <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">Metabolic Liability (DDI)</div>
                                                                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${admetData.cyp.overall_ddi_risk === 'Low' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
                                                                        {admetData.cyp.overall_ddi_risk} Risk
                                                                    </span>
                                                                </div>
                                                                <div className="grid grid-cols-3 gap-2">
                                                                    {Object.entries(admetData.cyp.isoforms).map(([isoform, data]) => (
                                                                        <div key={isoform} className="text-center p-1.5 bg-slate-50 rounded border border-slate-100">
                                                                            <div className="text-[10px] text-slate-500 font-medium">{isoform}</div>
                                                                            <div className={`text-xs font-bold ${data.inhibition_risk === 'High' ? 'text-red-500' : 'text-slate-700'}`}>
                                                                                {data.inhibition_risk === 'High' ? 'Inhibitor' : '-'}
                                                                            </div>
                                                                        </div>
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        )}

                                                    </div>
                                                </div>

                                                {/* External Links */}
                                                <div className="bg-indigo-50 border border-indigo-100 p-4 rounded-xl">
                                                    <h4 className="text-indigo-900 font-bold text-sm mb-2">Deep Dive Analysis</h4>
                                                    <p className="text-indigo-700 text-xs mb-3">Open this compound in specialized toxicology tools:</p>
                                                    <div className="flex flex-wrap gap-2">
                                                        {Object.values(admetData.admet_links).slice(0, 3).map((link, i) => (
                                                            <a
                                                                key={i}
                                                                href={link.url}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                className="px-3 py-1.5 bg-white text-indigo-600 text-xs font-bold rounded shadow-sm hover:bg-slate-50 transition-colors"
                                                            >
                                                                {link.name} ↗
                                                            </a>
                                                        ))}
                                                    </div>

                                                </div>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="text-center text-slate-400 mt-20">
                                            Select a ligand to analyze.
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                    </div>
                </div>

            </div>
        </div>
    )
}
