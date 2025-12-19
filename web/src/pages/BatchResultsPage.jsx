import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'
import MoleculeViewer from '../components/MoleculeViewer'
import { ChevronLeft, Download, Eye, Maximize2, RefreshCw, BarChart2, Star, Zap, Activity } from 'lucide-react'

export default function BatchResultsPage() {
    const { batchId } = useParams()
    const [batchData, setBatchData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [sortConfig, setSortConfig] = useState({ key: 'binding_affinity', direction: 'ascending' })

    // Viewer State
    const [firstJobPdbqt, setFirstJobPdbqt] = useState(null)
    const [firstJobReceptor, setFirstJobReceptor] = useState(null)
    const [firstJobId, setFirstJobId] = useState(null)
    const [firstJobName, setFirstJobName] = useState('')

    // Auto-Refresh
    useEffect(() => {
        if (batchId) fetchBatchDetails(batchId)
    }, [batchId])

    useEffect(() => {
        const interval = setInterval(() => {
            if (batchData && ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING'].includes(batchData.status)) {
                fetchBatchDetails(batchId, true)
            }
        }, 5000)
        return () => clearInterval(interval)
    }, [batchId, batchData?.status])

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
                    setFirstJobId(bestJob.id)
                    setFirstJobName(bestJob.ligand_filename)
                    fetchJobStructure(bestJob.id, session.access_token)
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

            // Fetch Files
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

    const handleJobSelect = async (job) => {
        if (job.status !== 'SUCCEEDED') return
        setFirstJobId(job.id)
        setFirstJobName(job.ligand_filename)
        const { data: { session } } = await supabase.auth.getSession()
        if (session) fetchJobStructure(job.id, session.access_token)
    }

    const handleSort = (key) => {
        let direction = 'ascending'
        if (sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending'
        }
        setSortConfig({ key, direction })
    }

    // Sort Logic
    const sortedJobs = batchData?.jobs ? [...batchData.jobs].sort((a, b) => {
        // Handle nulls
        const valA = a[sortConfig.key]
        const valB = b[sortConfig.key]

        // Always put nulls last
        if (valA === null) return 1
        if (valB === null) return -1

        if (valA < valB) return sortConfig.direction === 'ascending' ? -1 : 1
        if (valA > valB) return sortConfig.direction === 'ascending' ? 1 : -1
        return 0
    }) : []

    // Heatmap Helper
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
                <RefreshCw className="w-8 h-8 text-indigo-600 animate-spin" />
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
                        <ChevronLeft className="w-5 h-5" />
                    </Link>
                    <div>
                        <h1 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                            Batch Analysis <span className="px-2 py-0.5 rounded text-xs bg-slate-100 text-slate-500 font-mono">{batchId.slice(0, 8)}</span>
                        </h1>
                        <div className="text-xs text-slate-500 flex items-center gap-2">
                            {['RUNNING', 'SUBMITTED'].includes(batchData.status) ? (
                                <span className="flex items-center gap-1 text-blue-600 font-bold"><RefreshCw className="w-3 h-3 animate-spin" /> Processing ({batchData.stats.completed}/{batchData.stats.total})</span>
                            ) : (
                                <span className="text-emerald-600 font-bold flex items-center gap-1"><Star className="w-3 h-3 fill-current" /> Complete</span>
                            )}
                            <span className="text-slate-300">|</span>
                            <span>{batchData.stats.total} Ligands</span>
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    <button onClick={downloadCSV} className="btn-secondary btn-sm flex items-center gap-2">
                        <Download className="w-4 h-4" /> Export CSV
                    </button>
                    {/* Placeholder for PDF Report */}
                    <button disabled className="btn-secondary btn-sm flex items-center gap-2 opacity-50 cursor-not-allowed">
                        <BarChart2 className="w-4 h-4" /> PDF Report
                    </button>
                </div>
            </div>

            {/* 2. Main Workbench Area */}
            <div className="flex-1 flex overflow-hidden">

                {/* LEFT: Data Table with Heatmap */}
                <div className="w-1/3 min-w-[400px] border-r border-slate-200 bg-white flex flex-col">
                    <div className="p-4 border-b border-slate-200 bg-slate-50 flex justify-between items-center">
                        <h3 className="font-bold text-slate-700 text-sm uppercase tracking-wider">Results Table</h3>
                        <div className="text-xs text-slate-500">Sorted by Affinity</div>
                    </div>
                    <div className="flex-1 overflow-auto">
                        <table className="w-full text-sm text-left">
                            <thead className="text-xs text-slate-500 uppercase bg-slate-50 sticky top-0 z-10 shadow-sm">
                                <tr>
                                    <th onClick={() => handleSort('ligand_filename')} className="px-4 py-3 cursor-pointer hover:bg-slate-100">Ligand</th>
                                    <th onClick={() => handleSort('binding_affinity')} className="px-4 py-3 cursor-pointer hover:bg-slate-100 text-right">Affinity (kcal/mol)</th>
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
                                        <td className="px-4 py-3 text-right font-mono">
                                            <span className={getAffinityColor(job.binding_affinity)}>
                                                {job.binding_affinity ? job.binding_affinity.toFixed(1) : '-'}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* RIGHT: 3D Visualization */}
                <div className="flex-1 bg-slate-100 relative flex flex-col">
                    {/* Toolbar */}
                    <div className="absolute top-4 left-4 right-4 z-10 flex justify-between items-start pointer-events-none">
                        <div className="bg-white/90 backdrop-blur shadow-lg rounded-xl p-4 border border-slate-200 pointer-events-auto">
                            <h2 className="font-bold text-slate-900 mb-1">{firstJobName ? firstJobName.replace('.pdbqt', '') : 'Select a Ligand'}</h2>
                            <div className="text-xs text-slate-500 flex items-center gap-2">
                                <Activity className="w-3 h-3 text-indigo-500" /> Visualization
                            </div>
                        </div>
                        <div className="flex gap-2 pointer-events-auto">
                            <button className="p-2 bg-white shadow rounded-lg hover:bg-slate-50 text-slate-600" title="Reset View">
                                <Maximize2 className="w-5 h-5" />
                            </button>
                            <button className="p-2 bg-white shadow rounded-lg hover:bg-slate-50 text-slate-600" title="Style Toggle">
                                <Eye className="w-5 h-5" />
                            </button>
                        </div>
                    </div>

                    {/* Viewer Container */}
                    <div className="flex-1 w-full h-full relative">
                        {firstJobPdbqt ? (
                            <MoleculeViewer
                                pdbqtData={firstJobPdbqt}
                                receptorData={firstJobReceptor}
                                width="100%" // Fill parent flex
                                height="100%" // Fill parent flex
                                title="" // Hide default title
                            />
                        ) : (
                            <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400">
                                <Zap className="w-16 h-16 mb-4 opacity-50" />
                                <p className="text-lg font-medium">Select a ligand from the table to visualize</p>
                            </div>
                        )}
                    </div>
                </div>

            </div>
        </div>
    )
}
