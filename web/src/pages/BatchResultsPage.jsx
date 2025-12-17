import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import { API_URL } from '../config'

export default function BatchResultsPage() {
    const { jobId: batchId } = useParams() // Note: In App.jsx we might map it to :batchId, but let's check.
    // Actually standard is :batchId usually. Let's assume standard routing first.
    // Wait, the route in App.jsx hasn't been added yet. I will name the param "batchId".

    // Oh wait, I need to fetch the param correctly.
    // I'll assume the route I add in App.jsx is /dock/batch/:batchId

    const [batchData, setBatchData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [sortConfig, setSortConfig] = useState({ key: 'binding_affinity', direction: 'ascending' })

    const routingParams = useParams()
    const finalBatchId = routingParams.batchId || routingParams.jobId // Fallback if accidentally routed as jobId


    useEffect(() => {
        if (finalBatchId) {
            fetchBatchDetails(finalBatchId)
        }
    }, [finalBatchId])

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
        } catch (err) {
            console.error(err)
            setError(err.message)
        } finally {
            setLoading(false)
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

        // Header
        const headers = ['Ligand Name', 'Status', 'Binding Affinity (kcal/mol)', 'Job ID']

        // Rows
        const rows = batchData.jobs.map(job => [
            job.ligand_filename.replace('.pdbqt', ''),
            job.status,
            job.binding_affinity || 'N/A',
            job.id
        ])

        // Combine
        const csvContent = [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n')

        // Download
        const blob = new Blob([csvContent], { type: 'text/csv' })
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `batch_results_${batchData.batch_id}.csv`
        a.click()
        window.URL.revokeObjectURL(url)
    }

    const printReport = () => {
        window.print()
    }

    if (loading) return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50">
            <div className="text-center">
                <div className="text-4xl mb-4">🧪</div>
                <div className="text-slate-500">Loading Batch Results...</div>
            </div>
        </div>
    )

    if (error) return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50">
            <div className="text-center">
                <div className="text-4xl mb-4">⚠️</div>
                <h2 className="text-xl font-bold text-slate-900">Error Loading Batch</h2>
                <p className="text-slate-500 mt-2">{error}</p>
                <Link to="/dashboard" className="btn-secondary mt-4 inline-flex">Return to Dashboard</Link>
            </div>
        </div>
    )

    if (!batchData) return null

    return (
        <div className="min-h-screen bg-slate-50 pb-20">
            {/* Header */}
            <div className="bg-white border-b border-slate-200 pt-24 pb-12">
                <div className="container mx-auto px-4">
                    <div className="flex items-center gap-2 text-sm text-slate-500 mb-4">
                        <Link to="/dashboard" className="hover:text-primary-600">Dashboard</Link>
                        <span>/</span>
                        <span>Batch Results</span>
                    </div>

                    <div className="flex justify-between items-start">
                        <div>
                            <h1 className="text-3xl font-bold text-slate-900 mb-2">Batch Analysis</h1>
                            <p className="font-mono text-slate-500 text-sm">{batchData.batch_id}</p>
                        </div>
                        <div className="flex gap-3">
                            <button onClick={downloadCSV} className="btn-secondary flex items-center gap-2">
                                <span>📄</span> Download CSV
                            </button>
                            <button onClick={printReport} className="btn-secondary flex items-center gap-2">
                                <span>🖨️</span> Print PDF
                            </button>
                            <button onClick={() => fetchBatchDetails(finalBatchId)} className="btn-secondary">
                                🔄 Refresh
                            </button>
                        </div>
                    </div>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-8">
                        <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
                            <div className="text-sm text-slate-500 mb-1">Total Jobs</div>
                            <div className="text-2xl font-bold text-slate-900">{batchData.stats.total}</div>
                        </div>
                        <div className="bg-green-50 rounded-xl p-4 border border-green-100">
                            <div className="text-sm text-green-600 mb-1">Completed</div>
                            <div className="text-2xl font-bold text-green-700">{batchData.stats.completed}</div>
                        </div>
                        <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
                            <div className="text-sm text-blue-600 mb-1">Best Affinity</div>
                            <div className="text-2xl font-bold text-blue-700">
                                {batchData.stats.best_affinity?.toFixed(1) || '-'} <span className="text-sm font-normal">kcal/mol</span>
                            </div>
                        </div>
                        <div className="bg-purple-50 rounded-xl p-4 border border-purple-100">
                            <div className="text-sm text-purple-600 mb-1">Success Rate</div>
                            <div className="text-2xl font-bold text-purple-700">
                                {batchData.stats.success_rate.toFixed(0)}%
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Results Table */}
            <div className="container mx-auto px-4 py-8">
                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                    <div className="px-6 py-4 border-b border-slate-200 bg-slate-50 flex justify-between items-center">
                        <h3 className="font-bold text-slate-700">Individual Ligand Results</h3>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-slate-200">
                            <thead className="bg-slate-50">
                                <tr>
                                    <th
                                        onClick={() => handleSort('ligand_filename')}
                                        className="px-6 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider cursor-pointer hover:bg-slate-100"
                                    >
                                        Ligand Name {sortConfig.key === 'ligand_filename' && (sortConfig.direction === 'ascending' ? '↑' : '↓')}
                                    </th>
                                    <th
                                        onClick={() => handleSort('status')}
                                        className="px-6 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider cursor-pointer hover:bg-slate-100"
                                    >
                                        Status {sortConfig.key === 'status' && (sortConfig.direction === 'ascending' ? '↑' : '↓')}
                                    </th>
                                    <th
                                        onClick={() => handleSort('binding_affinity')}
                                        className="px-6 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider cursor-pointer hover:bg-slate-100"
                                    >
                                        Affinity (kcal/mol) {sortConfig.key === 'binding_affinity' && (sortConfig.direction === 'ascending' ? '↑' : '↓')}
                                    </th>
                                    <th className="px-6 py-3 text-right text-xs font-bold text-slate-500 uppercase tracking-wider">
                                        Actions
                                    </th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-slate-200">
                                {sortedJobs.map((job) => (
                                    <tr key={job.id} className="hover:bg-slate-50 transition-colors">
                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">
                                            {job.ligand_filename.replace('.pdbqt', '')}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                                            <span className={`px-2 py-1 text-xs rounded-full font-bold
                                                ${job.status === 'SUCCEEDED' ? 'bg-green-100 text-green-700' :
                                                    job.status === 'FAILED' ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'}
                                            `}>
                                                {job.status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700 font-bold">
                                            {job.binding_affinity ? job.binding_affinity.toFixed(1) : '-'}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                            <Link to={`/dock/${job.id}`} className="text-primary-600 hover:text-primary-800 mr-4">
                                                View Details
                                            </Link>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    )
}
