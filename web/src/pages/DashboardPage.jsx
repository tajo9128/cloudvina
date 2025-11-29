import { useState, useEffect } from 'react'
import { supabase } from '../supabaseClient'
import { Link } from 'react-router-dom'
import JobFilters from '../components/JobFilters'
import { API_URL } from '../config'

export default function DashboardPage() {
    const [jobs, setJobs] = useState([])
    const [loading, setLoading] = useState(true)
    const [filters, setFilters] = useState({
        status: '',
        search: '',
        minAffinity: '',
        maxAffinity: ''
    })

    useEffect(() => {
        fetchJobs()
    }, [filters])

    const handleFilterChange = (newFilters) => {
        setFilters(prev => ({ ...prev, ...newFilters }))
    }

    const fetchJobs = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) return

            const params = new URLSearchParams()
            if (filters.status) params.append('status', filters.status)
            if (filters.search) params.append('search', filters.search)
            if (filters.minAffinity) params.append('min_affinity', filters.minAffinity)
            if (filters.maxAffinity) params.append('max_affinity', filters.maxAffinity)

            const url = `${API_URL}/jobs?${params.toString()}`

            const response = await fetch(url, {
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                }
            })

            if (!response.ok) throw new Error('Failed to fetch jobs')

            const data = await response.json()
            setJobs(data)
        } catch (error) {
            console.error('Error fetching jobs:', error)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-blue-mesh">
            <div className="container mx-auto px-4 py-8">
                <div className="flex justify-between items-center mb-8">
                    <div>
                        <h1 className="text-3xl font-extrabold text-white tracking-tight">Dashboard</h1>
                        <p className="text-blue-200">Manage your docking jobs</p>
                    </div>
                    <Link to="/dock/new" className="btn-cyan px-6 py-2 rounded-xl text-sm font-bold shadow-lg shadow-cyan-500/20">
                        + New Job
                    </Link>
                </div>

                {/* Job Filters */}
                <JobFilters onFilterChange={handleFilterChange} />

                {loading ? (
                    <div className="text-center py-12">
                        <div className="text-white">
                            <div className="text-5xl mb-4 animate-bounce">🧪</div>
                            <div className="text-xl font-light text-blue-200">Loading jobs...</div>
                        </div>
                    </div>
                ) : jobs.length === 0 ? (
                    <div className="glass-modern p-12 text-center rounded-2xl">
                        <div className="text-4xl mb-4">🧪</div>
                        <h3 className="text-lg font-bold text-white mb-2">No jobs yet</h3>
                        <p className="text-blue-200 mb-6">Start your first molecular docking job now.</p>
                        <Link to="/dock/new" className="text-cyan-400 font-bold hover:text-cyan-300">
                            Create Job →
                        </Link>
                    </div>
                ) : (
                    <div className="glass-modern rounded-2xl overflow-hidden">
                        <table className="min-w-full divide-y divide-blue-800/30">
                            <thead className="bg-blue-900/40">
                                <tr>
                                    <th className="px-6 py-4 text-left text-xs font-bold text-blue-200 uppercase tracking-wider">Job ID</th>
                                    <th className="px-6 py-4 text-left text-xs font-bold text-blue-200 uppercase tracking-wider">Status</th>
                                    <th className="px-6 py-4 text-left text-xs font-bold text-blue-200 uppercase tracking-wider">Created</th>
                                    <th className="px-6 py-4 text-left text-xs font-bold text-blue-200 uppercase tracking-wider">Affinity</th>
                                    <th className="px-6 py-4 text-right text-xs font-bold text-blue-200 uppercase tracking-wider">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-blue-800/30">
                                {jobs.map((job) => (
                                    <tr key={job.id} className="hover:bg-white/5 transition-colors">
                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                                            <Link to={`/dock/${job.id}`} className="hover:text-cyan-400 transition-colors font-mono">
                                                {job.id.slice(0, 8)}...
                                            </Link>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                                            <span className={`px-3 py-1 inline-flex text-xs leading-5 font-bold rounded-full 
                                        ${job.status === 'SUCCEEDED' ? 'bg-green-500/20 text-green-300 border border-green-500/30' :
                                                    job.status === 'FAILED' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
                                                        'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'}`}>
                                                {job.status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-200">
                                            {new Date(job.created_at).toLocaleDateString()}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-white font-bold">
                                            {job.binding_affinity ? (
                                                <span className="text-cyan-300">{job.binding_affinity} kcal/mol</span>
                                            ) : (
                                                <span className="text-blue-400/50">-</span>
                                            )}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                            <Link to={`/dock/${job.id}`} className="text-cyan-400 hover:text-cyan-300 font-bold">
                                                View
                                            </Link>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    )
}
